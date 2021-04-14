use std::io::BufRead;
use std::sync::mpsc::{sync_channel, Receiver};
use std::{fmt, fs::File};

use ndarray::{s, Array2};
use numpy::{IntoPyArray, PyArray2};
use punkt_stable::{params, SentenceTokenizer, TrainingData};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use sentencepiece::SentencePieceProcessor;
use serde_json::Value;

#[derive(Debug)]
pub struct Lines<B> {
    buf: B,
}
pub trait LinesWithEnding<B> {
    fn lines_with_ending(self) -> Lines<B>;
}
impl<B> LinesWithEnding<B> for B
where
    B: BufRead,
{
    fn lines_with_ending(self) -> Lines<B> {
        Lines::<B> { buf: self }
    }
}
impl<B: BufRead> Iterator for Lines<B> {
    type Item = std::io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = String::new();
        match self.buf.read_line(&mut buf) {
            Ok(0) => None,
            Ok(_n) => Some(Ok(buf)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Infinite iterator over input files, in random order.
struct InputFileIterator {
    input_paths: Vec<String>,
    epoch_paths: Vec<String>,
}

impl InputFileIterator {
    fn new(input_paths: Vec<String>) -> InputFileIterator {
        let length = input_paths.len();
        InputFileIterator {
            input_paths,
            epoch_paths: Vec::with_capacity(length),
        }
    }
}

impl Iterator for InputFileIterator {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.epoch_paths.is_empty() {
            self.epoch_paths.extend_from_slice(&self.input_paths[..]);
            self.epoch_paths.shuffle(&mut rand::thread_rng());
        }
        self.epoch_paths.pop()
    }
}

thread_local! {
    static PUNKT_DATA: Box<TrainingData> = Box::new(TrainingData::english());
}

/// Shared code between .jsonl and .jsonl.zst files
fn documents_from_jsonl_reader<R>(
    reader: std::io::BufReader<R>,
) -> impl Iterator<Item = Vec<String>>
where
    R: std::io::Read,
{
    let json_deserializer = serde_json::Deserializer::from_reader(reader);
    let stream = json_deserializer.into_iter::<Value>();
    let documents = stream.filter_map(|val| {
        let mapping = match val {
            Ok(Value::Object(obj)) => Some(obj),
            _ => None,
        };
        let text = match mapping?.remove("text") {
            Some(Value::String(x)) => Some(x),
            _ => None,
        }?;
        let doc = PUNKT_DATA.with(|punkt_data| {
            let tokenized = SentenceTokenizer::<params::Standard>::new(&text, punkt_data);
            tokenized
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        });
        Some(doc)
    });
    documents
}

/// Read documents from a `.jsonl.zst` file
fn read_jsonl_zstd_documents(path: &str) -> impl Iterator<Item = Vec<String>> {
    let f = File::open(path).expect("failed to open data file");
    let decoder = zstd::stream::read::Decoder::new(f).expect("failed to create zstd decoder");
    let reader = std::io::BufReader::new(decoder);
    documents_from_jsonl_reader(reader)
}

/// Read documents from a `.jsonl` file
fn read_jsonl_documents(path: &str) -> impl Iterator<Item = Vec<String>> {
    let f = File::open(path).expect("failed to open data file");
    let reader = std::io::BufReader::new(f);
    documents_from_jsonl_reader(reader)
}

/// Read from a text file where documents are separated by a blank line
fn read_text_documents(path: &str) -> impl Iterator<Item = Vec<String>> {
    let f = File::open(path).expect("failed to open data file");
    let reader = std::io::BufReader::new(f);
    let mut sentences: Vec<String> = Vec::new();
    reader
        .lines_with_ending()
        .chain(std::iter::once(Ok(String::new())))
        .filter_map(move |line| {
            let line = line.expect("failed to read line from file");
            if line.trim().len() == 0 {
                let doc = sentences.split_off(0);
                assert!(sentences.is_empty());
                if !doc.is_empty() {
                    Some(doc)
                } else {
                    None
                }
            } else {
                sentences.push(line);
                None
            }
        })
}

/// Read documents from a file, dispatching based on file extension
fn read_documents(path: &str) -> Box<dyn Iterator<Item = Vec<String>> + Send> {
    match std::path::Path::new(&path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
    {
        Some("zst") => Box::new(read_jsonl_zstd_documents(path)),
        Some("jsonl") => Box::new(read_jsonl_documents(path)),
        _ => Box::new(read_text_documents(path)),
    }
}

#[derive(Debug)]
struct DocumentEncoding {
    input_ids: Vec<u32>,
    segment_ends: Vec<usize>,
}

#[derive(Debug)]
struct TokenizedDocument {
    encoding: DocumentEncoding,
    segment_cursor: usize,
}

impl TokenizedDocument {
    fn new(encoding: DocumentEncoding) -> TokenizedDocument {
        TokenizedDocument {
            encoding,
            segment_cursor: 0,
        }
    }

    fn next_segment_range(
        &mut self,
        length: usize,
        allow_segment_pairs: bool,
    ) -> Option<(std::ops::Range<usize>, Option<std::ops::Range<usize>>)> {
        if self.segment_cursor == self.encoding.segment_ends.len() {
            return None;
        }

        let mut start = if self.segment_cursor == 0 {
            0
        } else {
            self.encoding.segment_ends[self.segment_cursor - 1]
        };
        let mut used_segment_ends: Vec<usize> = Vec::new();
        let mut end = start;

        while end - start < length - 1 && self.segment_cursor < self.encoding.segment_ends.len() {
            // length - 1 to account for SEP token
            end = self.encoding.segment_ends[self.segment_cursor];
            used_segment_ends.push(end);
            self.segment_cursor += 1;
        }

        if !allow_segment_pairs || used_segment_ends.len() <= 1 {
            loop {
                let length_single = end - start;
                if length_single <= length {
                    break;
                }

                let truncate_from_front = rand::thread_rng().gen_bool(0.5);

                assert!(length_single > 0);
                if truncate_from_front {
                    start += 1;
                } else {
                    end -= 1;
                }
            }
            let segment_a = start..end;
            Some((segment_a, None))
        } else {
            let mut start_a = start;
            let mut end_b = end;

            let mut end_a =
                used_segment_ends[rand::thread_rng().gen_range(0..used_segment_ends.len() - 1)];
            let mut start_b = end_a;

            loop {
                let length_a = end_a - start_a;
                let length_b = end_b - start_b;
                if length_a + length_b <= length - 1 {
                    // length - 1 to account for SEP token
                    break;
                }

                let truncate_from_front = rand::thread_rng().gen_bool(0.5);
                if length_a > length_b {
                    assert!(length_a > 0);
                    if truncate_from_front {
                        start_a += 1;
                    } else {
                        end_a -= 1;
                    }
                } else {
                    assert!(length_b > 0);
                    if truncate_from_front {
                        start_b += 1;
                    } else {
                        end_b -= 1;
                    }
                }
            }
            let segment_a = start_a..end_a;
            let segment_b = start_b..end_b;
            Some((segment_a, Some(segment_b)))
        }
    }
}
#[pyclass(unsendable)]
struct InputPipeline {
    batch_size: usize,
    shuffle_buffer_size: usize,
    prefetch_amount: usize,
    sep_token_id: u32,
    pad_token_id: u32,
    receiver: Receiver<DocumentEncoding>,
    shuffle_buffer: Vec<TokenizedDocument>,
    active_documents: Vec<TokenizedDocument>,
}

impl fmt::Debug for InputPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InputPipeline")
            .field("batch_size", &self.batch_size)
            .field("shuffle_buffer_size", &self.shuffle_buffer_size)
            .field("prefetch_amount", &self.prefetch_amount)
            .finish()
    }
}

impl InputPipeline {
    fn get_document(
        shuffle_buffer: &mut Vec<TokenizedDocument>,
        shuffle_buffer_size: usize,
        prefetch_amount: usize,
        receiver: &Receiver<DocumentEncoding>,
    ) -> TokenizedDocument {
        while shuffle_buffer.len() < shuffle_buffer_size + 1 {
            let encoding = receiver.recv().expect(
                "failed to receive tokenized document in main thread, while filling shuffle buffer",
            );
            shuffle_buffer.push(TokenizedDocument::new(encoding));
        }

        while shuffle_buffer.len() < shuffle_buffer_size + prefetch_amount {
            let maybe_encoding = receiver.try_recv();
            match maybe_encoding {
                Ok(encoding) => shuffle_buffer.push(TokenizedDocument::new(encoding)),
                Err(_) => break,
            }
        }

        let i = rand::thread_rng().gen_range(0..shuffle_buffer_size);
        let doc = shuffle_buffer.pop().unwrap();
        std::mem::replace(&mut shuffle_buffer[i], doc)
    }

    fn write_example(
        batched_examples: &mut Array2<u32>,
        example_index: usize,
        tokenized_doc: &mut TokenizedDocument,
        length: usize,
        sep_token_id: u32,
        pad_token_id: u32,
        shuffle_buffer: &mut Vec<TokenizedDocument>,
        shuffle_buffer_size: usize,
        prefetch_amount: usize,
        receiver: &Receiver<DocumentEncoding>,
    ) {
        let mut segment_range = tokenized_doc.next_segment_range(length, true);
        while segment_range.is_none() {
            *tokenized_doc = InputPipeline::get_document(
                shuffle_buffer,
                shuffle_buffer_size,
                prefetch_amount,
                receiver,
            );
            segment_range = tokenized_doc.next_segment_range(length, true);
        }
        let (segment_range_a, maybe_segment_range_b) = segment_range.unwrap();

        let segment_length_a = segment_range_a.len();
        let segment_a = &tokenized_doc.encoding.input_ids[segment_range_a];
        let mut batch_slice_a = batched_examples.slice_mut(s![example_index, ..segment_length_a]);
        batch_slice_a.assign(&ndarray::ArrayView1::from(segment_a));

        match maybe_segment_range_b {
            Some(segment_range_b) => {
                let segment_length_b = segment_range_b.len();
                let segment_b = &tokenized_doc.encoding.input_ids[segment_range_b];

                batched_examples[(example_index, segment_length_a)] = sep_token_id;
                let mut batch_slice_b = batched_examples.slice_mut(s![
                    example_index,
                    segment_length_a + 1..segment_length_a + 1 + segment_length_b
                ]);
                batch_slice_b.assign(&ndarray::ArrayView1::from(segment_b));
                batched_examples
                    .slice_mut(s![example_index, segment_length_a + 1 + segment_length_b..])
                    .fill(pad_token_id);
            }
            None => {
                batched_examples
                    .slice_mut(s![example_index, segment_length_a..])
                    .fill(pad_token_id);
            }
        }
    }
}

#[pymethods]
impl InputPipeline {
    #[new]
    fn new(
        model_path: &str,
        batch_size: usize,
        input_paths: Vec<String>,
        shuffle_buffer_size: usize,
    ) -> InputPipeline {
        let prefetch_amount = 2 * batch_size;
        let (sender, receiver) = sync_channel::<DocumentEncoding>(prefetch_amount);

        let tokenizer = SentencePieceProcessor::open(model_path).unwrap();
        let sep_token_id = tokenizer.eos_id().unwrap();
        let pad_token_id = tokenizer
            .piece_to_id("<pad>")
            .unwrap()
            .expect("tokenizer has no padding token");

        std::thread::spawn(move || {
            let document_iter = InputFileIterator::new(input_paths)
                .flat_map(|path| read_documents(&path))
                .par_bridge();
            let encoded_iter = document_iter.filter_map(|doc| {
                let mut encoding = DocumentEncoding {
                    input_ids: Vec::new(),
                    segment_ends: Vec::new(),
                };
                for sentence in doc {
                    let encoded_sentence = tokenizer.encode(&sentence).unwrap();
                    if encoded_sentence.len() == 0 {
                        continue;
                    }
                    encoding.input_ids.extend(
                        encoded_sentence
                            .into_iter()
                            .map(|p| p.id)
                            .collect::<Vec<_>>(),
                    );
                    encoding.segment_ends.push(encoding.input_ids.len());
                }

                // Filter out empty documents right away, so they don't take up
                // space in any downnstream channel/prefetch/shuffle buffers.
                if encoding.input_ids.len() > 0 {
                    assert_ne!(encoding.segment_ends.len(), 0);
                    Some(encoding)
                } else {
                    None
                }
            });
            let out = encoded_iter.try_for_each_with(sender, |s, x| s.send(x));
            if let Err(_) = out {
                println!("[shutting down tokenization workers]")
            };
        });
        InputPipeline {
            batch_size,
            shuffle_buffer_size,
            prefetch_amount,
            sep_token_id,
            pad_token_id,
            receiver,
            shuffle_buffer: Vec::with_capacity(shuffle_buffer_size + prefetch_amount),
            active_documents: Vec::with_capacity(batch_size),
        }
    }

    fn get_batch<'py>(&mut self, py: Python<'py>, length: usize) -> &'py PyArray2<u32> {
        while self.active_documents.len() < self.batch_size {
            self.active_documents.push(InputPipeline::get_document(
                &mut self.shuffle_buffer,
                self.shuffle_buffer_size,
                self.prefetch_amount,
                &self.receiver,
            ));
        }

        let mut batch: Array2<u32> = Array2::zeros((self.batch_size, length));

        for (i, tokenized_doc) in self.active_documents.iter_mut().enumerate() {
            InputPipeline::write_example(
                &mut batch,
                i,
                tokenized_doc,
                length,
                self.sep_token_id,
                self.pad_token_id,
                &mut self.shuffle_buffer,
                self.shuffle_buffer_size,
                self.prefetch_amount,
                &self.receiver,
            );
        }
        batch.into_pyarray(py)
    }
}

#[pymodule]
fn sabertooth_pipeline(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<InputPipeline>()?;
    Ok(())
}
