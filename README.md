# Sabertooth

Sabertooth is standalone pre-training recipe based on JAX+Flax, with data pipelines implemented in Rust. It runs on CPU, GPU, and/or TPU, but this README targets TPU usage.


## Contents
1. [Installation](#installation)
2. [Data preparation](#data-preparation)
3. [Tokenizer preparation](#tokenizer-preparation)
4. [Pre-training](#pre-training)
5. [Fine-tuning on GLUE](#fine-tuning-on-glue)


## Installation

### Automatic installation in TPU VMs

This TPU code is intended to work with JAX TPU VMs as documented in the [JAX on Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).

Rather than creating VMs manually and installing dependencies by hand, you can use the scripts in `tpu_management` to help with this.

Before using these scripts:
- Follow the one-time setup instructions in the [JAX on Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm) up to (but not including) the point where you actually create a TPU VM.
- Then follow the instructions in `tpu_management/config.env.template` to set configuration variables used by the TPU management scripts.

After that, the following TPU management commands are available:
- `tpu_management/tpu.sh create NAME` creates a single-host TPU VM with the specified NAME and prints an ssh config for accessing the VM. Adding this config entry to `~/.ssh/config` will then allow you to use `ssh NAME` to access the VM. 
- `tpu_management/tpu.sh provision NAME` will push this git repository to the created VM, and build/install all required software within the VM. It also sets up NAME as a git remote, so you can do `git push NAME ...` to push code to the VM, and `git fetch NAME` to fetch commits developed within the VM.
- `tpu_management/tpu.sh delete NAME` deletes the TPU instance. All files on the TPU VM filesystem will be lost.
- `tpu_management/tpu.sh config-ssh NAME` prints the SSH config you need to connect to the VM. This command is useful when aVM automatically restarts and gets assigned a new IP address.

The `tpu_management/pod.sh` behaves similarly, but for multi-host TPU training:
- `tpu_management/pod.sh create NAME` creates a multi-host TPU setup with specified NAME, and sets up ssh config for each worker (`${NAME}0`, `${NAME}1`, ...).
- `tpu_management/pod.sh provision NAME` will configure all worker VMs, as well as setting up a git such that `git push NAME ...` will push to all four workers
- `tpu_management/tpu.sh delete NAME` deletes the TPU instance, including all workers. All files on the worker VM filesystems will be lost.
- `tpu_management/tpu.sh config-ssh NAME` prints the SSH config you need to connect to the workers. This command is useful when a VM automatically restarts and gets assigned a new IP address.

***Note: on a brand new TPU VM, doing `import tensorflow` for the first time can take minutes. This issue goes away on all subsequent calls. If a pre-training/fine-tuning script appears to hang on a new VM, this is probably the cause. You can test this by waiting out `python3 -c "import tensorflow"` and then running the actual script you want.***

### Manual installation

For Python dependencies, see `requirements_tpu.txt`. The input pipeline and data processing scripts are implemented in Rust. See https://rustup.rs/ for one-line shell command that installs Rust.

Run `./install_sabertooth_pipeline.sh` to build and install the `sabertooth_pipeline` helper package into the currently active python environment. This requires CMake to be installed.


## Data preparation

### Accepted formats

Sabertooth accepts pre-training data in either of the following formats:
* Text format with one sentence per line, with blank lines in between documents (the BERT format). With this format, sentence segmentation is assumed to have been fully carried out during pre-processing.
* [JSONlines](https://jsonlines.org/) format, which is automatically used for all files ending in `.jsonl`. A zstandard-compressed version is also accepted, which is used for all files ending in `.jsonl.zst` or plain `.zst`. Each line match the schema `{"text": "[JSON-encoded text of an entire document...]"}` (the Pile format). With this format, sentence segmentation will be performed at training time by background CPU threads that are also responsible for tokenization.

#### Pre-processed downloads

If you don't want to run the processing commands described in the next subsection, you can skip ahead by downloading already processed data from one of the following sources:
* [English Wikipedia in BERT format (4.5GB compressed download)](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/enwiki-197b5d8d.zip), courtesy of GluonNLP. The processing scripts in this repository are largely identical to the [GluonNLP processing scripts](https://github.com/dmlc/gluon-nlp/tree/master/scripts/datasets/pretrain_corpus), except that ours are implemented in Rust. The only downside of this download is that it does not include any Books data (potentially resulting in a lower GLUE score after pre-training), and does not shuffle the full corpus.
* [The Pile](https://pile.eleuther.ai/): 825GB of text from diverse sources, but some of this data is not quite as clean as our Wikipedia processing. Training BERT-base on the Pile consistently achieves a GLUE test score above 77, but we have not nailed the right set of hyperparameters for matching the effectiveness of Wiki+Books data.

### Preparing wikibooks data

#### Downloading and extracting: Wikipedia

First, download the [Wikipedia dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2) and extract the pages. The Wikipedia dump can be downloaded from this link, and should contain the following file: `enwiki-latest-pages-articles-multistream.xml.bz2`. Note that older database dumps are periodically deleted from the official Wikimedia downloads website, so we can't pin a specific version of Wikipedia without hosting a mirror of the full dataset.

Next, run WikiExtractor (`rust/create_pretraining_data/WikiExtractor.py`) to extract the wiki pages from the XML. The generated wiki pages file will be stored as `<data dir>/LL/wiki_nn`; for example `<data dir>/AA/wiki_00`. Each file is ~1MB, and each sub directory has 100 files from `wiki_00` to `wiki_99`, except the last sub directory. For the dump around December 2020, the last file is `FL/wiki_09`.

```sh
DATA_ROOT="$HOME/prep_sabertooth"
mkdir -p $DATA_ROOT
cd $DATA_ROOT
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2    # Optionally use curl instead
bzip2 -d enwiki-latest-pages-articles-multistream.xml.bz2
python3 rust/create_pretraining_data/WikiExtractor.py enwiki-latest-pages-articles-multistream.xml    # Results are placed in text/
```

#### Downloading and extracting: Books

Download and extract [books1](https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz).

#### Preprocessing and sharding

```sh
DATA_ROOT="$HOME/prep_sabertooth"
mkdir -p $DATA_ROOT/wikibooks
cd rust/create_pretraining_data
cargo build --release  # Removing `--release` will make the code *much* slower
target/release/create_pretraining_data --output $DATA_ROOT/wikibooks --num-shards 500 --wiki $DATA_ROOT/'text/??/wiki_??' --books $DATA_ROOT/'books1/epubtxt/*.txt'
```

This will write 500 shards in JSONlines format to the folder `wikibooks/`; each should be around 40MB in size. Set the `RAYON_NUM_THREADS` environment variable to limit the number of parallel threads used for processing; by default, a thread will be spawned per CPU core.

`create_pretraining_data` will load the full dataset into RAM to perform a global shuffle. If you run out of memory, you can it multiple times with a subset of the files. `--wiki` and `--books` both accept glob patterns, so you can do e.g. `--wiki 'text/A?/wiki_??'`. The `--wiki` or `--books` can also be repeated (with different files each time), or omitted. The JSONlines format should also make it easy to manually concatenate/split/shuffle shards of data.

#### Compressing the data

Compressing the data is optional, but it can save space when storing and transferring large datasets. All of our code supports uncompressed `.jsonl` and zstandard-compressed `.jsonl.zst` files interchangeably, so you can also skip this step (and change `.jsonl.zst` to `.jsonl` in all commands below).

To compress: download [zstandard](https://github.com/facebook/zstd/releases/), compile it with `make`, and run a compression command such as `zstd $DATA_ROOT/wikibooks/*.jsonl`.

The overhead of decompressing data is negligible when compared to more costly operations like tokenization or sentence segmentation, which we will be doing at training time anyway. The only downside of compressing the data is that you can't inspect compressed files with a simple text viewer/editor.


## Tokenizer preparation

We use [SentencePiece](https://github.com/google/sentencepiece) for tokenization.

For large datasets, feeding the raw data directly to the sentencepiece trainer is very slow in memory-hungry (RAM usage is at least 10x the size of the uncompressed raw text, and there are single-threaded O(n) sections in the sentencepiece trainer code).

Instead, we will first count unique words in our data, separated only by whitespace. We will then pass a TSV file containing the counts to the sentencepiece trainer, which will further decompose them and build a subword vocabulary.

To count the tokens in our data, run:
```sh
DATA_ROOT="$HOME/prep_sabertooth"
pushd rust/count_tokens
cargo build --release
popd
rust/count_tokens/target/release/count_tokens $DATA_ROOT/wikibooks/*.jsonl.zst > $DATA_ROOT/counts_wikibooks.tsv
```

`count_tokens` accepts filenames as arguments, and the supports the following formats:
- `.jsonl`: JSONlines data, where each line matches the schema `{"text": "[JSON-encoded text of an entire document...]"}`
- `.jsonl.zst` and `.zst`: same as above, but zstandard-compressed
- `.tsv`: merges counts from another tsv file into the output. If your available CPU RAM is not sufficient to count your corpus all at once, you can count shards of the data separately and then merge the tsv files
- All other extensions are treated as plain-text data

Once the TSV file is created, use our provided script to train the tokenizer:

```sh
python3 rust/count_tokens/train_tokenizer.py --input $DATA_ROOT/counts_wikibooks.tsv --model_prefix $DATA_ROOT/wikibooks_32k --vocab_size 32128
```

After this you're ready to run pre-training!

*Tip for data transfer*: The data folder (with zstandard-compressed shards) can be packed into a tar archive using the command `cd $DATA_DIR/.. && tar cvf prep_sabertooth.tar prep_sabertooth/wikibooks/*.jsonl.zst prep_sabertooth/*.model prep_sabertooth/*.vocab`. If you have multiple TPU VMs (either multiple workers for multi-host training, or just multiple VMs for different jobs), `scp`-ing data from one VM to another is 10x-100x faster than copying via `gsutil cp` or from any non-gcloud machine. SSH into a TPU VM and run `ifconfig` to determine its interal IP address that can be accessed by other VMs (typically of the form `10.x.y.z`).


## Pre-training

For single-host pre-training, run a command such as:
```sh
python3 run_pretraining.py --config=configs/pretraining.py --config.train_batch_size=1024 --config.optimizer="adam" --config.learning_rate=1e-4 --config.num_train_steps=1000000 --config.num_warmup_steps=10000 --config.max_seq_length=128 --config.max_predictions_per_seq=20
```

For multi-host training on `TPUv3-32` with 4 hosts, set up each host with the required environment variables and then run a command such as:
```sh
python3 run_pretraining.py --config=configs/pretraining.py --config.optimizer="adam" --config.train_batch_size=4096 --config.learning_rate=1e-3 --config.num_train_steps=125000 --config.num_warmup_steps=3125 --config.adam_epsilon=1e-11 --config.adam_beta1=0.9 --config.adam_beta2=0.98 --config.weight_decay=0.1 --config.max_grad_norm=0.4
```

Use `--config.input_files` and `--config.tokenizer` to configure dataset and tokenizer paths for pre-training (see `configs/pretraining.py` for the full set of configuration options and hyperparameters.)

### Pre-training notes

Our pre-training recipe is close to BERT, but there are a few differences:
* We use a SentencePiece unigram tokenizer, instead of WordPiece
* The next-sentence prediction (NSP) task from BERT is replaced with a sentence order prediction (SOP) from ALBERT. We do this primarily to simplify the data pipeline implementation, but past work has observed SOP to give better results than NSP.
* BERT's Adam optimizer departs from the Adam paper in that it omits bias correction terms. This codebase uses Optax's implementation of Adam, which includes bias correction.
* Pre-training uses a fixed maximum sequence length of 128, and does not increase the sequence length to 512 for the last 10% of training.
* The wiki+books data used in this repository is designed to match the BERT paper as closely as possible, but it's not identical. The data used by BERT was never publicly available, so most BERT replications have this property.
* Random masking and sentence shuffling occurs each time a batch of examples is sampled during training, rather than a single time during the data generation step.


## Fine-tuning on GLUE

Sample command for fine-tuning on GLUE:

```sh
python3 run_classifier.py --config=configs/classifier.py --config.init_checkpoint="/path/to/checkpoint/folder/" --config.dataset_name="cola" --config.learning_rate="5e-5"
```

The `dataset_name` should be one of: `cola`, `mrpc`, `qqp`, `sst2`, `stsb`, `mnli`, `qnli`, `rte`. WNLI is not supported because BERT accuracy on WNLI is below the baseline, unless a special training recipe is used.

### Leaderboard evaluation

To evaluate a model on GLUE, we typically run a sweep across different learning rates and use the development set to select the best one:
```sh
OUTPUT_DIR="$HOME/glue"
./sweep_glue.sh /path/to/checkpoint/folder $OUTPUT_DIR 5e-5 4e-5 3e-5 2e-5
```

Here are the results from one such sweep, using a "base" size model trained with batch size 1024 for 1M steps (see the single-host training command above). Our understanding is that this option most closely approximates the total compute resources used to train the original BERT-base, except that we do not increase the sequence length to 512 at any point during training. The learning rates the sweep found are random to some extent, since the sweep doubles as both a learning rate search and a chance to try different random seeds.

|      | CoLA | SST-2 | MRPC (f1/a) | STS-B (p/s) | QQP (f1/acc) | MNLI (m/mm) | QNLI | RTE  |
|------|------|-------|-------------|-------------|--------------|-------------|------|------|
| dev  | 56.5 | 91.9  | 90.8 / 87.3 | 88.3 / 88.4 |  87.0 / 90.4 | 84.9 / 85.5 | 92.1 | 70.4 |
| test | 57.0 | 92.5  | 87.3 / 82.8 | 87.8 / 87.0 |  71.3 / 89.1 | 85.1 / 84.3 | 92.1 | 66.7 |
| lr   | 3e-5 | 3e-5  |        3e-5 |        5e-5 |         3e-5 |        2e-5 | 3e-5 | 5e-5 |

Here are the results from another "base" size model, this time trained with batch size 4096 for 125K steps (see the multi-host training command above). Note that model only sees half the number of examples compared to the single-host training command above.

|      | CoLA | SST-2 | MRPC (f1/a) | STS-B (p/s) | QQP (f1/acc) | MNLI (m/mm) | QNLI | RTE  |
|------|------|-------|-------------|-------------|--------------|-------------|------|------|
| dev  | 60.3 | 91.9  | 89.6 / 85.5 | 87.9 / 88.1 |  87.2 / 90.6 | 84.7 / 85.0 | 91.7 | 68.6 |
| test | 51.9 | 92.1  | 88.8 / 84.7 | 86.2 / 85.0 |  70.9 / 89.0 | 84.5 / 84.0 | 91.5 | 65.9 |
| lr   | 4e-5 | 4e-5  |        5e-5 |        2e-5 |         4e-5 |        3e-5 | 3e-5 | 5e-5 |

