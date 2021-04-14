use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::Mutex,
};

use qp_trie::Trie;
use rayon::prelude::*;
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

// Shared code between .jsonl and .jsonl.zst files
fn documents_from_jsonl_reader<R>(reader: std::io::BufReader<R>) -> impl Iterator<Item = String>
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
        let mut mapping = mapping?;
        let maybe_pile_set_name = mapping
            .get("meta")
            .and_then(|meta| meta.get("pile_set_name"));
        match maybe_pile_set_name {
            Some(Value::String(pile_set_name)) => match pile_set_name.as_str() {
                // "ArXiv" => {},          // Maybe OK, but definitely lots of latex
                "BookCorpus2" => {}    // Good, but trace amounts of Greek/Russion
                "Books3" => {}         // Good, but some Polish/German/Greek
                "DM Mathematics" => {} // Mostly OK, but lots of Greek and Russion
                "Enron Emails" => {}   // Good, but does include Russian and Greek
                "FreeLaw" => {}        // Good, but again not all English
                // "Github" => {},            // Might be OK, suprisingly
                "Gutenberg (PG-19)" => {} // Good, but again not all English
                "HackerNews" => {}        // Surprisingly OK, but again not all English
                "NIH ExPorter" => {}      // Good, but again not all English
                "OpenSubtitles" => {}     // Good, but again not all English
                "OpenWebText2" => {}      // OK or maybe even good
                "Pile-CC" => {}           // Messy but seems OK
                "PubMed Abstracts" => {}  // Good
                // "PubMed Central" => {},    // Lots of markup, I could see a reason to omit
                // "StackExchange" => {},     // Tons of source code, probably omit
                "USPTO Backgrounds" => {} // Good
                // "Ubuntu IRC" => {},        // Lots of noise, maybe leave out?
                "Wikipedia (en)" => {} // OK, though not the most clean version of wikipedia
                // "YoutubeSubtitles" => {},  // Decent but noisy
                _ => return None,
            },
            _ => {}
        }

        let text = match mapping.remove("text") {
            Some(Value::String(x)) => Some(x),
            _ => None,
        };
        text
    });
    documents
}

/// Read documents from a `.jsonl.zst` file
fn read_jsonl_zstd_documents(path: &str) -> impl Iterator<Item = String> {
    let f = File::open(path).expect("failed to open data file");
    let decoder = zstd::stream::read::Decoder::new(f).expect("failed to create zstd decoder");
    let reader = std::io::BufReader::new(decoder);
    documents_from_jsonl_reader(reader)
}

/// Read documents from a `.jsonl` file
fn read_jsonl_documents(path: &str) -> impl Iterator<Item = String> {
    let f = File::open(path).expect("failed to open data file");
    let reader = std::io::BufReader::new(f);
    documents_from_jsonl_reader(reader)
}

/// Read from a text file where documents are separated by a blank line
fn read_text_documents(path: &str) -> impl Iterator<Item = String> {
    let f = File::open(path).expect("failed to open data file");
    let reader = std::io::BufReader::new(f);
    let mut buf = String::new();
    reader
        .lines_with_ending()
        .chain(std::iter::once(Ok(String::new())))
        .filter_map(move |line| {
            let line = line.expect("failed to read line from file");
            if line.trim().len() == 0 {
                let doc = String::from(&buf[..]);
                buf.clear();
                if doc.len() > 0 {
                    Some(doc)
                } else {
                    None
                }
            } else {
                buf.push_str(&line);
                None
            }
        })
}

/// Read documents from a file, dispatching based on file extension
fn read_documents(path: &str) -> Box<dyn Iterator<Item = String> + Send> {
    match std::path::Path::new(&path)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
    {
        Some("zst") => Box::new(read_jsonl_zstd_documents(path)),
        Some("jsonl") => Box::new(read_jsonl_documents(path)),
        _ => Box::new(read_text_documents(path)),
    }
}

#[inline]
fn is_whitespace(c: &u8) -> bool {
    *c == b' ' || *c == b'\t' || *c == b'\n' || *c == b'\r'
}

#[inline]
fn is_not_empty(s: &&[u8]) -> bool {
    !s.is_empty()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_files: Vec<String> = Vec::from(&args[1..]);
    let num_input_files = input_files.len();

    if num_input_files == 0 {
        eprintln!("Usage: count_tokens file [file2 [...]] > out.tsv");
        std::process::exit(1);
    }
    for path in input_files.iter() {
        if !std::path::Path::new(&path).is_file() {
            eprintln!("Usage: count_tokens file [file2 [...]] > out.tsv");
            eprintln!("Not a file: {}", path);
            std::process::exit(2);
        }
    }

    let aggregated_words_mutex: Mutex<Trie<Vec<u8>, u64>> = Mutex::new(Trie::new());

    let dummy_results: Vec<()> = input_files
        .into_par_iter()
        .map(|path| {
            let path = &path;
            let extension = std::path::Path::new(path)
                .extension()
                .and_then(std::ffi::OsStr::to_str);
            match extension {
                Some("tsv") => {
                    let f = File::open(path).expect("failed to open tsv file");
                    let reader = BufReader::new(f);

                    let mut aggregated_words = aggregated_words_mutex.lock().unwrap();
                    eprintln!("{} [merging...]", path);
                    for line in reader.lines() {
                        let line = line.expect("failed to read line");
                        if line.is_empty() {
                            continue;
                        }
                        let word_and_count: Vec<&str> = line.split('\t').collect();
                        assert_eq!(word_and_count.len(), 2, "bad tsv file");
                        let word = word_and_count[0].as_bytes();
                        let count: u64 = word_and_count[1]
                            .parse()
                            .expect("bad count (could not parse as int)");
                        if let Some(aggregated_count) = aggregated_words.get_mut(word) {
                            *aggregated_count += count;
                        } else {
                            aggregated_words.insert(Vec::from(word), count);
                        }
                    }
                    eprintln!("{} [DONE]", path);
                }
                _ => {
                    eprintln!("{} [0/?]", path);
                    let documents: Vec<String> = read_documents(path).collect();

                    let mut words: Trie<&[u8], u64> = Trie::new();
                    for (i, document) in documents.iter().enumerate() {
                        if i % 25000 == 0 {
                            eprintln!("{} [{}/{}]", path, i, documents.len());
                        }
                        for word in document
                            .as_bytes()
                            .split(is_whitespace)
                            .filter(is_not_empty)
                        {
                            if let Some(count) = words.get_mut(word) {
                                *count += 1;
                            } else {
                                words.insert(word, 1);
                            }
                        }
                    }
                    eprintln!("{} [{}/{}]", path, documents.len(), documents.len());

                    let mut aggregated_words = aggregated_words_mutex.lock().unwrap();
                    for (word, count) in words.iter() {
                        if let Some(aggregated_count) = aggregated_words.get_mut(*word) {
                            *aggregated_count += count;
                        } else {
                            aggregated_words.insert(Vec::from(*word), *count);
                        }
                    }
                    eprintln!("{} [DONE]", path);
                }
            };
        })
        .collect();
    assert_eq!(num_input_files, dummy_results.len());

    eprintln!("Printing word counts to stdout...");
    let aggregated_words = aggregated_words_mutex.lock().unwrap();
    for (word, count) in aggregated_words.iter() {
        let word = (*word).to_owned();
        let word = String::from_utf8(word).unwrap();
        println!("{}\t{}", word, count);
    }
}
