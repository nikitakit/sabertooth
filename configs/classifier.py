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

import ml_collections
from ml_collections.config_dict import config_dict


def get_config(config_string=""):
    """Configuration for fine-tuning on GLUE."""
    config = ml_collections.ConfigDict(
        {
            # Dataset path or identifier for HuggingFace datasets library
            "dataset_path": "glue",
            # Defining the name of the dataset configuration
            "dataset_name": "cola",
            # Initial checkpoint (e.g. from a pre-trained BERT model).
            "init_checkpoint": config_string,
            # Whether to run training.
            "do_train": True,
            # Whether to run eval on the dev set.
            "do_eval": True,
            # Whether to run the model in inference mode on the test set.
            "do_predict": True,
            # Total batch size for training.
            "train_batch_size": 32,
            # Total batch size for eval.
            "eval_batch_size": 8,
            # Total batch size for predict.
            "predict_batch_size": 8,
            # Optimizer: either 'adam' or 'lamb
            "optimizer": "adam",
            # The base learning rate for Adam or LAMB.
            "learning_rate": 5e-5,
            # The beta1 parameter for Adam or LAMB
            "adam_beta1": 0.9,
            # The beta2 parameter for Adam or LAMB
            "adam_beta2": 0.999,
            # The epsilon parameter for Adam or LAMB
            "adam_epsilon": 1e-6,
            # Weight decay rate for all parameters except biases and layer norm scale
            "weight_decay": 0.01,
            # Maximum gradient norm (for gradient clipping)
            "max_grad_norm": 1.0,
            # Total number of training epochs to perform.
            "num_train_epochs": 3.0,
            # Proportion of training to perform linear learning rate warmup for.
            # E.g., 0.1 = 10% of training.
            "warmup_proportion": 0.1,
            # The maximum total input sequence length after tokenization.
            # Sequences longer than this will be truncated, and sequences shorter
            # than this will be padded.
            "max_seq_length": 128,
            # Model configuration parameters, to be loaded from the pre-trained
            # inital checkpoint.
            "model": config_dict.placeholder(ml_collections.ConfigDict),
        }
    )

    return config
