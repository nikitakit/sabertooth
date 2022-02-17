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

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

import datetime
import glob
import itertools
import json
import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from flax.training import checkpoints
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


def get_output_dir(config):
    """Get output directory location."""
    del config
    output_dir = FLAGS.output_dir
    if output_dir is None:
        output_name = "pretrain_{timestamp}".format(
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
                masked_lm_positions=dummy_input,
                deterministic=True,
            )

        variable_dict = jax.jit(initialize_model)()
        return variable_dict["params"]


def compute_pretraining_loss_and_metrics(apply_fn, variables, batch, rngs):
    """Compute cross-entropy loss for classification tasks."""
    metrics = apply_fn(
        variables,
        batch["input_ids"],
        batch["input_mask"],
        batch["token_type_ids"],
        batch["masked_lm_positions"],
        batch["masked_lm_ids"],
        batch["masked_lm_weights"],
        batch["next_sentence_label"],
        rngs=rngs,
    )
    return metrics["loss"], metrics


def compute_pretraining_stats(apply_fn, variables, batch):
    """Used for computing eval metrics during pre-training."""
    masked_lm_logits, next_sentence_logits = apply_fn(
        variables,
        batch["input_ids"],
        batch["input_mask"],
        batch["token_type_ids"],
        batch["masked_lm_positions"],
        deterministic=True,
    )
    stats = modeling.BertForPreTraining.compute_metrics(
        masked_lm_logits,
        next_sentence_logits,
        batch["masked_lm_ids"],
        batch["masked_lm_weights"],
        batch["next_sentence_label"],
    )

    masked_lm_correct = jnp.sum(
        (masked_lm_logits.argmax(-1) == batch["masked_lm_ids"].reshape((-1,)))
        * batch["masked_lm_weights"].reshape((-1,))
    )
    next_sentence_labels = batch["next_sentence_label"].reshape((-1,))
    next_sentence_correct = jnp.sum(
        next_sentence_logits.argmax(-1) == next_sentence_labels
    )
    stats = {
        "masked_lm_correct": masked_lm_correct,
        "masked_lm_total": jnp.sum(batch["masked_lm_weights"]),
        "next_sentence_correct": next_sentence_correct,
        "next_sentence_total": jnp.sum(jnp.ones_like(next_sentence_labels)),
        **stats,
    }
    return stats


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config = FLAGS.config

    input_files = sum([glob.glob(pattern) for pattern in config.input_files], [])
    assert input_files, "No input files!"
    print(f"Training with {len(input_files)} input files, including:")
    print(f" - {input_files[0]}")

    model = modeling.BertForPreTraining(config=config.model)
    initial_params = get_initial_params(model, init_checkpoint=config.init_checkpoint)
    tx = training.create_optimizer(
        optimizer=config.optimizer,
        b1=config.adam_beta1,
        b2=config.adam_beta2,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        learning_rate=config.learning_rate,
        warmup_steps=config.num_warmup_steps,
        total_steps=config.num_train_steps,
    )
    state = training.TrainState.create(
        apply_fn=model.apply,
        params=initial_params,
        tx=tx,
        train_rngs={"dropout": jax.random.PRNGKey(np.random.randint(2**16))},
        history=training.MetricHistory(),
    )
    del initial_params  # the state takes ownership of all params

    output_dir = get_output_dir(config)
    gfile.makedirs(output_dir)

    # Restore from a local checkpoint, if one exists.
    state = checkpoints.restore_checkpoint(output_dir, state)
    start_step = int(state.step)

    state = state.replicate()

    data_pipeline = data.PretrainingDataPipeline(
        sum([glob.glob(pattern) for pattern in config.input_files], []),
        config.tokenizer,
        max_seq_length=config.max_seq_length,
        max_predictions_per_seq=config.max_predictions_per_seq,
    )

    if config.do_train:
        train_batch_size = config.train_batch_size
        if jax.process_count() > 1:
            assert (
                train_batch_size % jax.process_count() == 0
            ), "train_batch_size must be divisible by number of processes"
            train_batch_size = train_batch_size // jax.process_count()
        train_iter = data_pipeline.get_inputs(
            batch_size=train_batch_size, training=True
        )
        train_step_fn = training.create_train_step(compute_pretraining_loss_and_metrics)

        for step, batch in zip(range(start_step, config.num_train_steps), train_iter):
            state = train_step_fn(state, batch)
            if jax.process_index() == 0 and (
                step % config.save_checkpoints_steps == 0
                or step == config.num_train_steps - 1
            ):
                checkpoints.save_checkpoint(output_dir, state.unreplicate(), step)
                config_path = os.path.join(output_dir, "config.json")
                if not os.path.exists(config_path):
                    with open(config_path, "w") as f:
                        json.dump({"model_type": "bert", **config.model}, f)
                tokenizer_path = os.path.join(output_dir, "sentencepiece.model")
                if not os.path.exists(tokenizer_path):
                    shutil.copy(config.tokenizer, tokenizer_path)

        # With the current Rust data pipeline code, running more than one pipeline
        # at a time will lead to a hang. A simple workaround is to fully delete the
        # training pipeline before potentially starting another for evaluation.
        del train_iter

    if config.do_eval:
        eval_iter = data_pipeline.get_inputs(batch_size=config.eval_batch_size)
        eval_iter = itertools.islice(eval_iter, config.max_eval_steps)
        eval_fn = training.create_eval_fn(
            compute_pretraining_stats, sample_feature_name="input_ids"
        )
        eval_stats = eval_fn(state, eval_iter)

        eval_metrics = {
            "loss": jnp.mean(eval_stats["loss"]),
            "masked_lm_loss": jnp.mean(eval_stats["masked_lm_loss"]),
            "next_sentence_loss": jnp.mean(eval_stats["next_sentence_loss"]),
            "masked_lm_accuracy": jnp.sum(eval_stats["masked_lm_correct"])
            / jnp.sum(eval_stats["masked_lm_total"]),
            "next_sentence_accuracy": jnp.sum(eval_stats["next_sentence_correct"])
            / jnp.sum(eval_stats["next_sentence_total"]),
        }

        eval_results = []
        for name, val in sorted(eval_metrics.items()):
            line = f"{name} = {val:.06f}"
            print(line, flush=True)
            eval_results.append(line)

        eval_results_path = os.path.join(output_dir, "eval_results.txt")
        with gfile.GFile(eval_results_path, "w") as f:
            for line in eval_results:
                f.write(line + "\n")


if __name__ == "__main__":
    app.run(main)
