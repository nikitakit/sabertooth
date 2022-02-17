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

"""Helper utilities for training."""

import time
from typing import Any, Dict, Optional

import jax
import numpy as np
import optax
from flax import jax_utils, struct
from flax.training import common_utils, train_state
from jax import numpy as jnp


class MetricHistory:
    """Container for reporting training metrics.

    Currently this only handles periodic logging, but the eventual design goal
    is to also allow this to store any metric history needed for training
    (e.g. for learning rate decay based on lack of improvement on a certain
    metric.)
    """

    def __init__(self, print_every=200):
        self.print_every = print_every
        self.last_printed = None
        self.estimated_step = 0

    def write(self, step, metrics):
        """TODO(kitaev): doc."""
        # The canonical value of the training step we're in (the "step" argument)
        # may be stored on TPU, in which case transfering to the CPU in order to
        # do a comparison would substantially slow down training.
        if self.estimated_step % self.print_every != 0:
            self.estimated_step += 1
            return
        if not isinstance(step, int):
            if step.ndim == 0:
                step = step.item()
            else:
                step = step[0].item()
        self.estimated_step = step

        # Only retrieve metrics from the device if they are actually used.
        metrics = jax.tree_map(
            lambda x: x[0].item() if x.ndim > 0 else x.item(), metrics
        )
        for i, k in enumerate(sorted(metrics)):
            if i == 0:
                line = f"Step {step-1:<7d} {k} = {metrics[k]}"
            else:
                line = f"             {k} = {metrics[k]}"
            print(line, flush=True)

        now = time.time()
        if self.last_printed:
            last_step, last_time = self.last_printed
            seconds_per_step = (now - last_time) / (step - last_step)
            line = f"             seconds_per_step = {seconds_per_step}"
            print(line, flush=True)
        self.last_printed = (step, now)


class TrainState(train_state.TrainState):
    train_rngs: Optional[Dict[str, Any]]
    history: MetricHistory = struct.field(pytree_node=False)

    def replicate(self):
        train_rngs = jax.tree_map(common_utils.shard_prng_key, self.train_rngs)
        replicated = jax_utils.replicate(self)
        replicated = harmonize_across_hosts(replicated)
        replicated = replicated.replace(train_rngs=train_rngs)
        return replicated

    def unreplicate(self):
        return jax_utils.unreplicate(self)


def create_optimizer(
    optimizer,
    b1,
    b2,
    eps,
    weight_decay,
    max_grad_norm,
    learning_rate,
    warmup_steps,
    total_steps,
):
    tx_chain = []

    if max_grad_norm:
        tx_chain.append(optax.clip_by_global_norm(max_grad_norm))

    tx_chain.extend(
        [
            optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
            optax.add_decayed_weights(
                weight_decay, lambda p: jax.tree_map(lambda x: x.ndim != 1, p)
            ),
        ]
    )

    if optimizer == "adam":
        pass
    elif optimizer == "lamb":
        tx_chain.append(optax.scale_by_trust_ratio())
    else:
        raise ValueError("Unsupported value for optimizer: {optimizer}")

    schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
            ),
            optax.linear_schedule(
                init_value=learning_rate,
                end_value=0.0,
                transition_steps=total_steps - warmup_steps,
            ),
        ],
        [warmup_steps],
    )
    tx_chain.append(optax.scale_by_schedule(lambda count: -1 * schedule(count)))
    tx = optax.chain(*tx_chain)
    return tx


def create_train_step(loss_and_metrics_fn):
    def train_step(state, batch):
        train_rngs, rng_treedef = jax.tree_flatten(state.train_rngs)
        split_rngs = [jax.random.split(rng) for rng in train_rngs]
        step_rngs = jax.tree_unflatten(rng_treedef, [x[0] for x in split_rngs])
        new_train_rngs = jax.tree_unflatten(rng_treedef, [x[1] for x in split_rngs])

        grad_fn = jax.value_and_grad(
            lambda params: loss_and_metrics_fn(
                state.apply_fn, {"params": params}, batch, step_rngs
            ),
            has_aux=True,
        )
        (unused_loss, metrics), grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, "batch")
        new_state = state.apply_gradients(grads=grads, train_rngs=new_train_rngs)
        return new_state, metrics

    p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=(0,))

    def distributed_train_step(state, batch):
        new_state, metrics = p_train_step(state, common_utils.shard(batch))
        new_state.history.write(new_state.step, metrics)
        return new_state

    return distributed_train_step


def create_eval_fn(stat_fn, sample_feature_name="idx"):
    """Constructs a function that runs evaluation given a batched data stream."""
    p_stat_fn = jax.pmap(
        lambda state, batch: stat_fn(state.apply_fn, {"params": state.params}, batch),
        axis_name="batch",
    )
    n_devices = jax.local_device_count()

    def eval_step_fn(state, batch, all_stats):
        batch_size = batch[sample_feature_name].shape[0]
        remainder = batch_size % n_devices
        if remainder:
            pad_amount = n_devices - remainder

            def pad(x):
                assert x.shape[0] == batch_size
                return np.concatenate([x] + [x[:1]] * pad_amount, axis=0)

            batch = jax.tree_map(pad, batch)
        batch = common_utils.shard(batch)
        stats = p_stat_fn(state, batch)
        stats = jax.tree_map(np.array, stats)
        stats = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), stats)
        if remainder:
            stats = jax.tree_map(lambda x: x[:-pad_amount], stats)
        all_stats.append(stats)

    def eval_fn(state, data_stream):
        all_stats = []
        for batch in data_stream:
            eval_step_fn(state, batch, all_stats)
        res = {}
        for k in all_stats[0]:
            res[k] = np.concatenate([stats[k] for stats in all_stats], axis=0)
        return res

    return eval_fn


def harmonize_across_hosts(optimizer):
    """Ensure that model and optimizer parameters are identical for all hosts."""
    if jax.process_count() == 1:
        return optimizer
    else:
        selector = jnp.zeros(jax.local_device_count())
        if jax.process_index() == 0:
            selector = jax.ops.index_update(selector, 0, 1.0)
        optimizer = jax.pmap(
            lambda opt, sel: jax.tree_map(
                lambda x: jax.lax.psum(x * sel.astype(x.dtype), "i"), opt
            ),
            axis_name="i",
        )(optimizer, selector)
        return optimizer
