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

"""Transformer models."""

import functools
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.checkpoints import restore_checkpoint
from flax.training.common_utils import onehot
from ml_collections import ConfigDict

import layers

ACT2FN = {
    "gelu": layers.gelu,
    "relu": nn.relu,
    "swish": nn.swish,
    "gelu_new": nn.gelu,
}


def get_hidden_activation(config: ConfigDict):
    return ACT2FN[config.hidden_act]


def get_kernel_init(config: ConfigDict):
    return layers.truncated_normal_initializer(config.initializer_range)


class BertModel(nn.Module):
    """BERT model without any task-specific heads."""

    config: ConfigDict

    def setup(self):
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=get_kernel_init(self.config),
            name="word_embeddings",
        )
        self.position_embeddings = layers.PositionalEncoding(
            num_embeddings=self.config.max_position_embeddings,
            features=self.config.hidden_size,
            embedding_init=get_kernel_init(self.config),
            name="position_embeddings",
        )
        self.type_embeddings = nn.Embed(
            num_embeddings=self.config.type_vocab_size,
            features=self.config.hidden_size,
            embedding_init=get_kernel_init(self.config),
            name="type_embeddings",
        )
        self.embeddings_layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, name="embeddings_layer_norm"
        )
        self.embeddings_dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

        build_feed_forward = functools.partial(
            layers.FeedForward,
            d_model=self.config.hidden_size,
            d_ff=self.config.intermediate_size,
            intermediate_activation=get_hidden_activation(self.config),
            kernel_init=get_kernel_init(self.config),
        )
        build_self_attention = functools.partial(
            layers.SelfAttention,
            num_heads=self.config.num_attention_heads,
            qkv_features=self.config.hidden_size,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=False,
            kernel_init=get_kernel_init(self.config),
            bias_init=nn.initializers.zeros,
        )
        self.encoder_layers = [
            layers.TransformerBlock(
                build_feed_forward=build_feed_forward,
                build_self_attention=build_self_attention,
                dropout_rate=self.config.hidden_dropout_prob,
                layer_norm_epsilon=self.config.layer_norm_eps,
                name=f"encoder_layer_{layer_num}",
            )
            for layer_num in range(self.config.num_hidden_layers)
        ]
        self.pooler = nn.Dense(
            kernel_init=get_kernel_init(self.config),
            name="pooler",
            features=self.config.hidden_size,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        input_mask: jnp.ndarray,
        type_ids: jnp.ndarray,
        *,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Applies BERT model on the inputs."""

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(input_ids)
        type_embeddings = self.type_embeddings(type_ids)

        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.embeddings_layer_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings, deterministic=deterministic)

        hidden_states = embeddings

        mask = input_mask.astype(jnp.int32)
        for transformer_block in self.encoder_layers:
            hidden_states = transformer_block(
                hidden_states, mask, deterministic=deterministic
            )
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = jnp.tanh(pooled_output)

        return hidden_states, pooled_output

    def get_embedding_table(self, **unused_kwargs):
        return self.variables["params"]["word_embeddings"]["embedding"]


class GatherIndexes(nn.Module):
    """Gathers the vectors at the specific positions."""

    @nn.compact
    def __call__(self, sequence_tensor: jnp.ndarray, positions: jnp.ndarray):
        """Applies gather indexes layer.
        Args:
          sequence_tensor: Sequence output of `BertModel` layer of shape
            (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
            hidden units of `BertModel` layer.
          positions: Positions ids of tokens in sequence to mask for pretraining
            of with dimension (batch_size, num_predictions) where
            `num_predictions` is maximum number of tokens to mask out and predict
            per each sequence.
        Returns:
          Masked out sequence tensor of shape (batch_size * num_predictions,
          num_hidden).
        """
        batch_size, seq_length, width = sequence_tensor.shape
        flat_offsets = jnp.reshape(jnp.arange(batch_size) * seq_length, [-1, 1])
        flat_positions = jnp.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = jnp.reshape(
            sequence_tensor, [batch_size * seq_length, width]
        )
        output_tensor = jnp.take(flat_sequence_tensor, flat_positions, axis=0)

        return output_tensor


class BertForSequenceClassification(nn.Module):
    """Bert model for sequence classification."""

    config: ConfigDict
    n_classes: int

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        input_mask: jnp.ndarray,
        type_ids: jnp.ndarray,
        labels: jnp.ndarray = None,
        *,
        deterministic: bool = False,
    ):
        """Applies BERT for sequence classification."""
        bert = BertModel(config=self.config, name="bert")
        _, pooled_output = bert(
            input_ids, input_mask, type_ids, deterministic=deterministic
        )
        pooled_output = nn.Dropout(
            rate=self.config.hidden_dropout_prob, deterministic=deterministic
        )(pooled_output)
        logits = layers.OutputProjection(
            n_out=self.n_classes,
            kernel_init=get_kernel_init(self.config),
            name="classification",
        )(pooled_output)

        if labels is None:
            return logits
        elif logits.shape[-1] == 1:
            # Regression task
            loss = jnp.mean((logits[..., 0] - labels) ** 2)
            return {"loss": loss}
        else:
            # Classification task
            logits = nn.log_softmax(logits)
            loss = -jnp.mean(
                jnp.sum(onehot(labels, logits.shape[-1]) * logits, axis=-1)
            )
            return {"loss": loss}

    @staticmethod
    def params_from_checkpoint(model, checkpoint):
        """Initialize params (but not optimizer) from a pre-trained checkpoint."""
        restored = restore_checkpoint(checkpoint, target=None)
        params = restored["target"]

        # Delete the masked lm head
        del params["predictions_output"]
        del params["predictions_transform_dense"]
        del params["predictions_transform_layernorm"]

        # Re-initialize the output head
        # If we switch to a non-static method, flax will complain that we're
        # creating an OutputProjection module here.
        params["classification"] = layers.OutputProjection(n_out=model.n_classes).init(
            jax.random.PRNGKey(np.random.randint(2 ** 16)),
            jnp.zeros(
                (1, model.config.hidden_size),
                dtype=params["classification"]["kernel"].dtype,
            ),
        )["params"]

        # Convert any numpy arrays (which live CPU) to JAX DeviceArrays
        params = jax.tree_map(jnp.asarray, params)
        # Always use a FrozenDict to store params. If different container types are
        # used at different places in the code, JAX may need to re-JIT model calls
        # because FrozenDict and python dict input structures are not identical.
        params = flax.core.freeze(params)
        return params


class BertForPreTraining(nn.Module):
    """Bert model for pre-training."""

    config: ConfigDict

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        input_mask: jnp.ndarray,
        type_ids: jnp.ndarray,
        masked_lm_positions: jnp.ndarray = None,
        masked_lm_labels: jnp.ndarray = None,
        masked_lm_weights: jnp.ndarray = None,
        next_sentence_labels: jnp.ndarray = None,
        *,
        deterministic: bool = False,
    ):
        """Applies BERT for pre-training."""
        config = self.config
        bert = BertModel(config=config, name="bert")
        sequence_output, pooled_output = bert(
            input_ids, input_mask, type_ids, deterministic=deterministic
        )
        if masked_lm_positions is None:
            return sequence_output, pooled_output

        # Masked LM
        masked_lm_input = GatherIndexes()(sequence_output, masked_lm_positions)
        masked_lm_input = nn.Dense(
            features=config.hidden_size,
            kernel_init=get_kernel_init(config),
            name="predictions_transform_dense",
        )(masked_lm_input)
        masked_lm_input = get_hidden_activation(config)(masked_lm_input)
        masked_lm_input = nn.LayerNorm(
            epsilon=config.layer_norm_eps, name="predictions_transform_layernorm"
        )(masked_lm_input)
        masked_lm_logits = layers.OutputProjection(name="predictions_output")(
            masked_lm_input, bert.get_embedding_table()
        )

        # Next-sentence prediction
        next_sentence_logits = layers.OutputProjection(
            n_out=2, kernel_init=get_kernel_init(config), name="classification"
        )(pooled_output)

        if masked_lm_labels is None or next_sentence_labels is None:
            return masked_lm_logits, next_sentence_logits
        else:
            return self.compute_metrics(
                masked_lm_logits,
                next_sentence_logits,
                masked_lm_labels,
                masked_lm_weights,
                next_sentence_labels,
            )

    @staticmethod
    def compute_metrics(
        masked_lm_logits: jnp.ndarray,
        next_sentence_logits: jnp.ndarray,
        masked_lm_labels: jnp.ndarray,
        masked_lm_weights: jnp.ndarray,
        next_sentence_labels: jnp.ndarray,
    ):
        """Computes the pre-training loss and its components."""
        masked_lm_logits = nn.log_softmax(masked_lm_logits)
        masked_lm_labels = onehot(
            masked_lm_labels.reshape((-1,)), masked_lm_logits.shape[-1]
        )
        masked_lm_weights = masked_lm_weights.reshape((-1,))
        masked_lm_loss = -jnp.sum(
            jnp.sum(masked_lm_logits * masked_lm_labels, axis=-1) * masked_lm_weights
        ) / jnp.sum(masked_lm_weights)

        next_sentence_logits = nn.log_softmax(next_sentence_logits)
        next_sentence_labels = next_sentence_labels.reshape((-1,))
        next_sentence_loss = -jnp.mean(
            jnp.sum(
                onehot(next_sentence_labels, next_sentence_logits.shape[-1])
                * next_sentence_logits,
                axis=-1,
            )
        )
        return {
            "loss": masked_lm_loss + next_sentence_loss,
            "masked_lm_loss": masked_lm_loss,
            "next_sentence_loss": next_sentence_loss,
        }

    @staticmethod
    def params_from_checkpoint(model, checkpoint):
        """Initialize params (but not optimizer) from a pre-trained checkpoint."""
        restored = restore_checkpoint(checkpoint, target=None)
        params = restored["target"]
        # Convert any numpy arrays (which live CPU) to JAX DeviceArrays
        params = jax.tree_map(jnp.asarray, params)
        # Always use a FrozenDict to store params. If different container types are
        # used at different places in the code, JAX may need to re-JIT model calls
        # because FrozenDict and python dict input structures are not identical.
        params = flax.core.freeze(params)
        return params
