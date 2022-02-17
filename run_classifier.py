# Copyright 2020 The Sabertooth Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run sequence-level classification (and regression) fine-tuning."""

import datetime
import logging
import os

import datasets
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import transformers
from absl import app, flags
from ml_collections.config_flags import config_flags
from tensorflow.io import gfile

import data
import modeling
import training

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory where the model checkpoints will be written.",
)

config_flags.DEFINE_config_file("config", None, "Hyperparameter configuration")


def get_config():
    config = FLAGS.config
    hf_config = transformers.AutoConfig.from_pretrained(config.init_checkpoint)
    assert hf_config.model_type == "bert", "Only BERT is supported."
    model_config = ml_collections.ConfigDict(
        {
            "vocab_size": hf_config.vocab_size,
            "hidden_size": hf_config.hidden_size,
            "num_hidden_layers": hf_config.num_hidden_layers,
            "num_attention_heads": hf_config.num_attention_heads,
            "hidden_act": hf_config.hidden_act,
            "intermediate_size": hf_config.intermediate_size,
            "hidden_dropout_prob": hf_config.hidden_dropout_prob,
            "attention_probs_dropout_prob": hf_config.attention_probs_dropout_prob,
            "max_position_embeddings": hf_config.max_position_embeddings,
            "type_vocab_size": hf_config.type_vocab_size,
            "initializer_range": hf_config.initializer_range,
            "layer_norm_eps": hf_config.layer_norm_eps,
        }
    )
    config.model = model_config
    return config


def get_output_dir(config):
    """Get output directory location."""
    output_dir = FLAGS.output_dir
    if output_dir is None:
        dataset_name = config.dataset_name.replace("/", "_")
        output_name = "{dataset_name}_{timestamp}".format(
            dataset_name=dataset_name,
            timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        )
        output_dir = os.path.join("~", "sabertooth", "output", output_name)
        output_dir = os.path.expanduser(output_dir)
        print()
        print("No --output_dir specified")
        print("Using default output_dir:", output_dir, flush=True)
    return output_dir


def get_initial_params(model, init_checkpoint=None):
    if init_checkpoint:
        return model.params_from_checkpoint(model, init_checkpoint)
    else:

        def initialize_model():
            dummy_input = jnp.zeros((1, 1), dtype=jnp.int32)
            return model.init(
                jax.random.PRNGKey(np.random.randint(2**16)),
                input_ids=dummy_input,
                input_mask=dummy_input,
                type_ids=dummy_input,
                labels=jnp.zeros(1, dtype=jnp.int32),
                deterministic=True,
            )

        variable_dict = jax.jit(initialize_model)()
        return variable_dict["params"]


def compute_loss_and_metrics(apply_fn, variables, batch, rngs):
    """Compute cross-entropy loss for classification tasks."""
    metrics = apply_fn(
        variables,
        batch["input_ids"],
        batch["input_mask"],
        batch["token_type_ids"],
        batch["label"],
        rngs=rngs,
    )
    return metrics["loss"], metrics


def compute_classification_stats(apply_fn, variables, batch):
    y = apply_fn(
        variables,
        batch["input_ids"],
        batch["input_mask"],
        batch["token_type_ids"],
        deterministic=True,
    )
    return {"idx": batch["idx"], "label": batch["label"], "prediction": y.argmax(-1)}


def compute_regression_stats(apply_fn, variables, batch):
    y = apply_fn(
        variables,
        batch["input_ids"],
        batch["input_mask"],
        batch["token_type_ids"],
        deterministic=True,
    )
    return {
        "idx": batch["idx"],
        "label": batch["label"],
        "prediction": y[..., 0],
    }


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config = get_config()

    datasets.logging.set_verbosity_error()
    # Workaround for https://github.com/huggingface/datasets/issues/812
    logging.getLogger("filelock").setLevel(logging.ERROR)
    dataset = datasets.load_dataset(config.dataset_path, config.dataset_name)
    data_pipeline = data.ClassificationDataPipeline(
        dataset, config.init_checkpoint, max_seq_length=config.max_seq_length
    )

    num_train_examples = len(dataset["train"])
    num_train_steps = int(
        num_train_examples * config.num_train_epochs // config.train_batch_size
    )
    warmup_steps = int(config.warmup_proportion * num_train_steps)

    is_regression_task = dataset["train"].features["label"].dtype == "float32"
    if is_regression_task:
        num_classes = 1
        compute_stats = compute_regression_stats
    else:
        num_classes = dataset["train"].features["label"].num_classes
        compute_stats = compute_classification_stats

    model = modeling.BertForSequenceClassification(
        config=config.model, n_classes=num_classes
    )
    initial_params = get_initial_params(model, init_checkpoint=config.init_checkpoint)
    tx = training.create_optimizer(
        optimizer=config.optimizer,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=num_train_steps,
    )
    state = training.TrainState.create(
        apply_fn=model.apply,
        params=initial_params,
        tx=tx,
        train_rngs={"dropout": jax.random.PRNGKey(np.random.randint(2**16))},
        history=training.MetricHistory(),
    )
    del initial_params  # the state takes ownership of all params
    state = state.replicate()

    output_dir = get_output_dir(config)
    gfile.makedirs(output_dir)

    if config.do_train:
        train_step_fn = training.create_train_step(compute_loss_and_metrics)
        train_iter = data_pipeline.get_inputs(
            split="train", batch_size=config.train_batch_size, training=True
        )

        for step, batch in zip(range(0, num_train_steps), train_iter):
            state = train_step_fn(state, batch)

    if config.do_eval:
        eval_fn = training.create_eval_fn(compute_stats)
        eval_results = []

        if config.dataset_path == "glue" and config.dataset_name == "mnli":
            validation_splits = ["validation_matched", "validation_mismatched"]
        else:
            validation_splits = ["validation"]

        for split in validation_splits:
            eval_iter = data_pipeline.get_inputs(
                split=split, batch_size=config.eval_batch_size, training=False
            )
            eval_stats = eval_fn(state, eval_iter)
            eval_metric = datasets.load_metric(config.dataset_path, config.dataset_name)
            eval_metric.add_batch(
                predictions=eval_stats["prediction"], references=eval_stats["label"]
            )
            eval_metrics = eval_metric.compute()
            prefix = "eval_mismatched" if split == "validation_mismatched" else "eval"
            for name, val in sorted(eval_metrics.items()):
                line = f"{prefix}_{name} = {val:.06f}"
                print(line, flush=True)
                eval_results.append(line)

        eval_results_path = os.path.join(output_dir, "eval_results.txt")
        with gfile.GFile(eval_results_path, "w") as f:
            for line in eval_results:
                f.write(line + "\n")

    if config.do_predict:
        predict_fn = training.create_eval_fn(compute_stats)

        path_map = {
            ("glue", "cola", "test"): "CoLA.tsv",
            ("glue", "mrpc", "test"): "MRPC.tsv",
            ("glue", "qqp", "test"): "QQP.tsv",
            ("glue", "sst2", "test"): "SST-2.tsv",
            ("glue", "stsb", "test"): "STS-B.tsv",
            ("glue", "mnli", "test_matched"): "MNLI-m.tsv",
            ("glue", "mnli", "test_mismatched"): "MNLI-mm.tsv",
            ("glue", "qnli", "test"): "QNLI.tsv",
            ("glue", "rte", "test"): "RTE.tsv",
            # No eval on WNLI for now. BERT accuracy on WNLI is below baseline,
            # unless a special training recipe is used.
            # ('glue/wnli', 'test'): 'WNLI.tsv',
        }
        label_sets = {
            ("glue", "cola"): ["0", "1"],
            ("glue", "mrpc"): ["0", "1"],
            ("glue", "qqp"): ["0", "1"],
            ("glue", "sst2"): ["0", "1"],
            ("glue", "mnli"): ["entailment", "neutral", "contradiction"],
            ("glue", "qnli"): ["entailment", "not_entailment"],
            ("glue", "rte"): ["entailment", "not_entailment"],
        }

        for path_map_key in path_map:
            candidate_dataset_path, candidate_dataset_name, split = path_map_key
            if (
                candidate_dataset_path != config.dataset_path
                or candidate_dataset_name != config.dataset_name
            ):
                continue

            predict_iter = data_pipeline.get_inputs(
                split=split, batch_size=config.eval_batch_size, training=False
            )
            predict_stats = predict_fn(state, predict_iter)
            idxs = predict_stats["idx"]
            predictions = predict_stats["prediction"]

            tsv_path = os.path.join(
                output_dir, path_map[config.dataset_path, config.dataset_name, split]
            )
            with gfile.GFile(tsv_path, "w") as f:
                f.write("index\tprediction\n")
                if is_regression_task:
                    for idx, val in zip(idxs, predictions):
                        f.write(f"{idx}\t{val:.06f}\n")
                else:
                    label_set = label_sets[config.dataset_path, config.dataset_name]
                    for idx, val in zip(idxs, predictions):
                        f.write(f"{idx}\t{label_set[val]}\n")
            print("Wrote", tsv_path)


if __name__ == "__main__":
    app.run(main)
