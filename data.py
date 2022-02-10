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

"""Input pipelines."""

import os

import numpy as np
import sabertooth_pipeline
import sentencepiece as spm
import torch


class DataPipeline:
    """Base class for input pipelines.

    Use get_inputs(...) to generate an iterator over batches of data, represented
    as python dicts mapping from strings to containing numpy arrays.
    """

    def get_inputs(self, batch_size, split=None, training=False):
        raise NotImplementedError("DataPipeline subclasses must define get_inputs.")


class HfDataPipeline(DataPipeline):
    def __init__(self, dataset):
        self.dataset = dataset

    def get_inputs(self, batch_size, split=None, training=False):
        dataloader = torch.utils.data.DataLoader(
            self.dataset if split is None else self.dataset[split],
            collate_fn=self.collate,
            batch_size=batch_size,
            drop_last=training,
            shuffle=training,
        )
        if training:
            while True:
                for batch in iter(dataloader):
                    yield dict(
                        batch
                    )  # The dict-like types from huggingface datasets are not pytrees
        else:
            for batch in iter(dataloader):
                yield dict(
                    batch
                )  # The dict-like types from huggingface datasets are not pytrees

    def collate(self, examples):
        raise NotImplementedError(
            "HfDataPipeline subclasess must define a collate function."
        )


class ClassificationDataPipeline(HfDataPipeline):
    def __init__(self, dataset, tokenizer_file, max_seq_length=128):
        if os.path.isdir(tokenizer_file):
            tokenizer_file = os.path.join(tokenizer_file, "sentencepiece.model")
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=tokenizer_file, add_bos=True, add_eos=True
        )
        self.pad_token_id = self.tokenizer.pad_id()
        self.max_seq_length = max_seq_length

        if isinstance(dataset, dict):
            single_split = dataset["train"]
        else:
            single_split = dataset

        self.name_a, *names_other = [
            name
            for name, feature in single_split.features.items()
            if feature.dtype == "string"
        ]
        assert (
            len(names_other) <= 1
        ), "Only single sentences and sentence pairs allowed."
        if names_other:
            self.name_b = names_other[0]
        else:
            self.name_b = None
        mapped_dataset = dataset.map(self.tokenize, batched=True)
        mapped_dataset.set_format(
            "numpy", columns=["idx", "input_ids", "token_type_ids", "label"]
        )
        super().__init__(mapped_dataset)

    def truncate_sequence_pair(self, ids_a, ids_b):
        num_tokens_to_remove = len(ids_a) + len(ids_b) + 3 - self.max_seq_length
        if num_tokens_to_remove <= 0:
            return ids_a, ids_b
        truncate_amount_a = 0
        truncate_amount_b = 0
        for _ in range(num_tokens_to_remove):
            length_a = len(ids_a) - truncate_amount_a
            length_b = len(ids_b) - truncate_amount_b
            if length_a > length_b:
                truncate_amount_a += 1
            else:
                truncate_amount_b += 1
        assert truncate_amount_a + truncate_amount_b == num_tokens_to_remove
        assert truncate_amount_a <= len(ids_a)
        assert truncate_amount_b <= len(ids_b)
        return ids_a[:-truncate_amount_a], ids_b[:-truncate_amount_b]

    def tokenize(self, examples):
        batch_input_ids = []
        batch_token_type_ids = []
        if self.name_b is not None:
            for text_a, text_b in zip(examples[self.name_a], examples[self.name_b]):
                ids_a = self.tokenizer.encode(text_a)
                ids_b = self.tokenizer.encode(text_b)
                assert len(ids_a) >= 2
                assert len(ids_b) >= 2
                cls_id = ids_a[0]
                sep_id1 = ids_a[-1]
                sep_id2 = ids_b[-1]
                ids_a, ids_b = self.truncate_sequence_pair(ids_a[1:-1], ids_b[1:-1])
                input_ids = [cls_id] + ids_a + [sep_id1] + ids_b + [sep_id2]
                assert len(input_ids) <= self.max_seq_length
                token_type_ids = [0 for _ in range(len(ids_a) + 2)] + [
                    1 for _ in range(len(ids_b) + 1)
                ]
                assert len(token_type_ids) <= self.max_seq_length
                batch_input_ids.append(input_ids)
                batch_token_type_ids.append(token_type_ids)
        else:
            for text_a in examples[self.name_a]:
                ids_a = self.tokenizer.encode(text_a)
                assert len(ids_a) >= 2
                num_tokens_to_truncate = max(0, len(ids_a) - self.max_seq_length)
                if num_tokens_to_truncate > 0:
                    input_ids = (
                        ids_a[:1] + ids_a[1 : -1 - num_tokens_to_truncate] + ids_a[-1:]
                    )
                else:
                    input_ids = ids_a
                assert len(input_ids) <= self.max_seq_length
                token_type_ids = [0 for _ in range(len(input_ids))]
                assert len(token_type_ids) <= self.max_seq_length
                batch_input_ids.append(input_ids)
                batch_token_type_ids.append(token_type_ids)

        return {
            **examples,
            "input_ids": batch_input_ids,
            "token_type_ids": batch_token_type_ids,
        }

    def collate(self, examples):
        idx = np.array([example["idx"] for example in examples], dtype=np.int32)
        label = np.array([example["label"] for example in examples], dtype=np.int32)
        input_ids = np.full(
            (len(examples), self.max_seq_length), self.pad_token_id, dtype=np.int32
        )
        token_type_ids = np.zeros((len(examples), self.max_seq_length), dtype=np.int32)
        for i, example in enumerate(examples):
            example_len = example["input_ids"].shape[0]
            input_ids[i, :example_len] = example["input_ids"]
            token_type_ids[i, :example_len] = example["token_type_ids"]
        return {
            "idx": idx,
            "label": label,
            "input_ids": input_ids,
            "input_mask": (input_ids != self.pad_token_id).astype(np.int32),
            "token_type_ids": token_type_ids,
        }


class PretrainingDataPipeline(DataPipeline):
    def __init__(
        self,
        input_files,
        tokenizer_file,
        max_seq_length=128,
        max_predictions_per_seq=20,
    ):
        super().__init__()
        self.input_files = input_files
        self.tokenizer_file = tokenizer_file
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_file)
        self.vocab_size = self.tokenizer.vocab_size()
        self.mask_token_id = self.tokenizer.piece_to_id("<mask>")
        self.cls_token_id = self.tokenizer.bos_id()
        self.sep_token_id = self.tokenizer.eos_id()
        self.pad_token_id = self.tokenizer.pad_id()

        self.ignore_ids = np.array(
            [self.cls_token_id, self.sep_token_id, self.pad_token_id], dtype=np.int64
        )[:, None, None]

    def get_inputs(self, batch_size, split=None, training=False):
        shuffle_buffer_size = 2048 if training else 1
        pipeline = sabertooth_pipeline.InputPipeline(
            self.tokenizer_file,
            batch_size,
            self.input_files,
            shuffle_buffer_size,
        )
        if training:
            while True:
                input_ids = pipeline.get_batch(self.max_seq_length - 2)
                yield self.process_batch({"input_ids": input_ids})
        else:
            for _ in range(4096 // batch_size):
                input_ids = pipeline.get_batch(self.max_seq_length - 2)
                yield self.process_batch({"input_ids": input_ids})

    def process_batch(self, batch):
        batch_size = batch["input_ids"].shape[0]
        batch["masked_lm_positions"] = np.zeros(
            (batch_size, self.max_predictions_per_seq), dtype=np.int64
        )
        batch["masked_lm_ids"] = np.zeros(
            (batch_size, self.max_predictions_per_seq), dtype=np.int64
        )
        batch["masked_lm_weights"] = np.zeros(
            (batch_size, self.max_predictions_per_seq), dtype=np.float32
        )

        # Add [CLS] and [SEP] tokens
        new_input_ids = np.full(
            (batch_size, self.max_seq_length), self.pad_token_id, dtype=np.int64
        )
        new_input_ids[:, 0] = self.cls_token_id
        new_input_ids[:, 1:-1] = batch["input_ids"]
        new_input_ids = np.where(
            np.cumsum(new_input_ids == self.pad_token_id, axis=-1) == 1,
            self.sep_token_id,
            new_input_ids,
        )
        batch["input_ids"] = new_input_ids

        # Sentence Order Prediction task
        # replace this block with the line below to disable SOP
        # batch['next_sentence_label'] = np.zeros(batch_size, dtype=np.int64)
        batch["next_sentence_label"] = np.random.randint(
            0, 2, batch_size, dtype=np.int64
        )
        batch["next_sentence_label"] = np.where(
            np.sum(batch["input_ids"] == self.sep_token_id, axis=-1) < 2,
            np.zeros_like(batch["next_sentence_label"]),
            batch["next_sentence_label"],
        )
        segments = np.cumsum(batch["input_ids"][:, ::-1] == self.sep_token_id, axis=-1)[
            :, ::-1
        ]
        segments[:, 0] = 1
        swapped_segments = np.argsort(
            np.where(segments == 1, -3, -segments), axis=-1, kind="stable"
        )
        swapped_input_ids = np.take_along_axis(
            batch["input_ids"], swapped_segments, axis=-1
        )
        batch["input_ids"] = np.where(
            batch["next_sentence_label"][:, None], swapped_input_ids, batch["input_ids"]
        )

        # Token type ids
        token_type_ids = np.cumsum(
            batch["input_ids"][:, ::-1] == self.sep_token_id, axis=-1
        )[:, ::-1]
        token_type_ids = (token_type_ids - token_type_ids[:, :1]) % 2
        batch["token_type_ids"] = token_type_ids

        # Masked LM task
        prediction_mask = np.all(batch["input_ids"] != self.ignore_ids, axis=0)
        num_tokens = np.sum(batch["input_ids"] != self.pad_token_id, axis=-1)
        for i in range(batch_size):
            cand_indexes = np.arange(prediction_mask.shape[1], dtype=np.int32)[
                prediction_mask[i]
            ]
            num_to_predict = min(
                self.max_predictions_per_seq, max(1, int(num_tokens[i] * 0.15))
            )

            masked_lm_positions = np.random.choice(
                cand_indexes, num_to_predict, replace=False
            )
            masked_lm_positions = np.sort(masked_lm_positions)
            masked_lm_ids = batch["input_ids"][i, masked_lm_positions]
            batch["masked_lm_positions"][i, :num_to_predict] = masked_lm_positions
            batch["masked_lm_ids"][i, :num_to_predict] = masked_lm_ids
            batch["masked_lm_weights"][i, :num_to_predict] = 1.0

        do_predict = prediction_mask[
            np.arange(batch_size)[:, None], batch["masked_lm_positions"]
        ]
        r = np.random.random(batch["masked_lm_ids"].shape)
        keep_original = (r < 0.1) | ~do_predict
        replace_with_mask = r < 0.9

        batch["input_ids"][
            np.arange(batch_size)[:, None], batch["masked_lm_positions"]
        ] = np.where(
            keep_original,
            # 10% of the time, keep original
            batch["input_ids"][
                np.arange(batch_size)[:, None], batch["masked_lm_positions"]
            ],
            np.where(
                replace_with_mask,
                # 80% of the time, replace with [MASK]
                self.mask_token_id,
                # 10% of the time, replace with random word
                np.random.randint(0, self.vocab_size, batch["masked_lm_ids"].shape),
            ),
        )

        batch["input_mask"] = (batch["input_ids"] != self.pad_token_id).astype(np.int32)

        return batch
