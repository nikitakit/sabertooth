use std::{
    fs::File,
    io::{BufRead, BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::{value_t, App, Arg};
use glob;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde_json::json;

/// Reads Wikipedia documents as outputted by WikiExtractor
fn read_extracted_wiki_documents(path: &Path) -> impl Iterator<Item = String> {
    let f = std::fs::File::open(path).expect("failed to open extracted wiki file");
    let reader = std::io::BufReader::new(f);

    let mut article_lines: Vec<String> = Vec::new();
    let mut article_open = false;

    reader.lines().filter_map(move |line| {
        let line = line.expect("failed to read line from file");

        if line.contains("<doc") {
            article_open = true;
            None
        } else if line.contains("</doc>") {
            article_open = false;
            // Skip line 0, which is just the title of the article
            let article = article_lines[1..].join("\n");
            article_lines.clear();
            Some(article)
        } else {
            if article_open && !line.is_empty() {
                article_lines.push(line);
            }
            None
        }
    })
}

/// Reads books in plaintext format (originally converted from epub)
fn read_extracted_books_documents(path: &Path) -> impl Iterator<Item = String> {
    let text = std::fs::read_to_string(path).expect("failed to read extracted book file");

    let mut lines: Vec<&str> = Vec::new();
    for line in text.lines() {
        // Skip empty lines and trim whitespace
        let line = line.trim();
        if !line.is_empty() {
            lines.push(line);
        }
    }

    let cleaned_text = lines.join("\n");
    // println!("{}", cleaned_text);
    std::iter::once(cleaned_text)
}

fn paths_from_patterns(patterns: Vec<&str>) -> Vec<std::path::PathBuf> {
    patterns
        .iter()
        .flat_map(|pattern| {
            let mut entries = glob::glob(pattern)
                .expect("Failed to read glob pattern")
                .peekable();
            if entries.peek().is_none() {
                eprintln!("ERROR: pattern did not match any files: {}", pattern);
                std::process::exit(1);
            }
            entries.map(|entry| {
                let entry = entry.expect("failed to read glob pattern");
                entry
            })
        })
        .collect()
}

fn get_output_paths(base_output_path: PathBuf, num_shards: usize) -> Vec<PathBuf> {
    let mut output_paths: Vec<PathBuf> = Vec::new();

    if num_shards == 0 {
        if base_output_path.exists() {
            eprintln!(
                "ERROR: {}: {}",
                if base_output_path.is_dir() {
                    "requested single-file output but output path is a directory"
                } else {
                    "output path already exists"
                },
                base_output_path.display()
            );
            std::process::exit(1);
        }
        output_paths.push(base_output_path);
    } else {
        if !base_output_path.is_dir() {
            eprintln!(
                "ERROR: requested multiple shards but output path is not a directory: {}",
                base_output_path.display()
            );
            std::process::exit(1);
        }
        for n in 0..num_shards {
            let shard_path =
                base_output_path.join(format!("part-{:05}-of-{:05}.jsonl", n, num_shards));
            if shard_path.exists() {
                eprintln!("ERROR: already exists: {}", shard_path.display());
                std::process::exit(1);
            }
            output_paths.push(shard_path);
        }
    }
    output_paths
}

fn main() {
    let matches = App::new("Data preparation for pre-training")
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("PATH")
                .help("Location to write the outputs")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("num-shards")
                .short("n")
                .long("num-shards")
                .value_name("NUMBER")
                .help("Number of output shards to write (set to 0 to write a single file instead)")
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::with_name("wiki")
                .short("w")
                .long("wiki")
                .value_name("PATTERN")
                .help("Reads files previously produced by WikiExtractor")
                .takes_value(true)
                .number_of_values(1)
                .multiple(true),
        )
        .arg(
            Arg::with_name("books")
                .short("b")
                .long("books")
                .alias("book")
                .value_name("PATTERN")
                .help("Reads text files previously extracted from epub")
                .takes_value(true)
                .number_of_values(1)
                .multiple(true),
        )
        .arg(
            Arg::with_name("no-shuffle")
                .long("no-shuffle")
                .help("Disables shuffling of output"),
        )
        .get_matches();

    let num_shards: usize = value_t!(matches, "num-shards", usize).unwrap_or(0);
    let shuffle = !matches.is_present("no-shuffle");
    let wiki_patterns: Vec<_> = matches.values_of("wiki").unwrap_or_default().collect();
    let books_patterns: Vec<_> = matches.values_of("books").unwrap_or_default().collect();

    let base_output_path = PathBuf::from(matches.value_of("output").unwrap());
    let output_paths = get_output_paths(base_output_path, num_shards);

    let wiki_iter = paths_from_patterns(wiki_patterns)
        .into_par_iter()
        .flat_map_iter(|path| {
            println!("Reading from {}...", path.display());
            read_extracted_wiki_documents(&path).map(|doc| {
                let json_doc = json!({
                    "text": doc,
                    "meta": {
                        "dataset": "Wikipedia",
                    }
                });
                json_doc.to_string()
            })
        });

    println!("Reading from books files...");
    let books_iter = paths_from_patterns(books_patterns)
        .into_par_iter()
        .flat_map_iter(|path| {
            read_extracted_books_documents(&path).map(|doc| {
                let json_doc = json!({
                    "text": doc,
                    "meta": {
                        "dataset": "Books",
                    }
                });
                json_doc.to_string()
            })
        });

    let mut documents: Vec<String> = wiki_iter.chain(books_iter).collect();
    if documents.is_empty() {
        println!("No documents to process. Exiting.");
        return;
    }

    if shuffle {
        println!("Shuffling all documents...");
        documents.shuffle(&mut rand::thread_rng());
        println!("Done shuffling.");
    }

    let mut documents_per_shard = documents.len() / output_paths.len();
    if documents_per_shard * num_shards < documents.len() {
        documents_per_shard += 1
    }

    documents
        .par_chunks(documents_per_shard)
        .enumerate()
        .for_each(|(shard_index, shard_documents)| {
            let shard_path = &output_paths[shard_index];
            let mut writer =
                BufWriter::new(File::create(shard_path).expect("Unable to create file"));
            println!(
                "Writing {} documents to {}...",
                shard_documents.len(),
                shard_path.display()
            );
            for doc in shard_documents {
                writer.write_all(&doc.as_bytes()).expect("Failed to write");
                writer.write_all(b"\n").expect("Failed to write");
            }
        });
}
