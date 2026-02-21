# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Transformer module."""

import haiku as hk
from haiku import initializers as init
import jax.numpy as jnp
import jax
from emergent_in_context_learning.modules import transformer_core


class LinearTransformer(hk.Module):
    """Transformer tower with Linear Attention."""

    def __init__(self,
                 input_embedder,
                 num_classes=1623,
                 num_layers=8,
                 num_heads=8,
                 dropout_prob=0.1,
                 self_att_init_scale=1.0,
                 dense_init_scale=1.0,
                 name=None):
        """Initialize the Transformer tower.

        Args:
          input_embedder: InputEmbedder object.
          num_classes: Total number of output classes.
          num_layers: Number of transformer blocks.
          num_heads: Number of transformer heads.
          dropout_prob: Dropout probability.
          self_att_init_scale: Scale for self-attention initialization.
          dense_init_scale: Scale for dense layer initialization.
          name: Optional name for the module.
        """
        super(LinearTransformer, self).__init__(name=name)
        self._input_embedder = input_embedder
        self._num_classes = num_classes
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_prob = dropout_prob
        self._self_att_init_scale = self_att_init_scale
        self._dense_init_scale = dense_init_scale

    def __call__(self,
                 examples,
                 labels,
                 mask=None,
                 is_training=True,
                 return_embeddings=False):
        """Call to the Transformer tower.

        Args:
          examples: input sequence of shape [batch_size, seq_len, ...].
          labels: input sequence of shape [batch_size, seq_len].
          mask: optional input mask of shape [batch_size, seq_len]. (0 for padding)
          is_training: if is currently training.

        Returns:
          outputs: output logits or embeddings.
        """
        # 1. Input Embeddings
        hh = self._input_embedder(examples, labels, is_training)

        # Project input to hidden dimension (assumed 64 based on original code,
        # but strictly speaking usually depends on head_dim * num_heads)
        # Using existing convention:
        hh = transformer_core.conv1(hh, 64, init_scale=self._dense_init_scale)

        # 2. Linear Transformer Layers
        for i in range(self._num_layers):
            # Linear Attention does not use the N*N attention matrix mask.
            # It only needs the [B, S] padding mask to zero out K/V.
            # Note: The original code multiplied `hh *= mask` manually.
            # We keep that if desired, but LayerNorm typically centers it back.
            # It's cleaner to let the attention mechanism handle the masking internally.

            if mask is not None:
                # Optional: Clean the residual stream at padding locations
                hh *= mask[:, :, None]

            hh = transformer_core.LinearTransformerBlock(
                num_heads=self._num_heads,
                widening_factor=4,
                dropout_prob=self._dropout_prob,
                self_att_init_scale=self._self_att_init_scale,
                dense_init_scale=self._dense_init_scale,
                name=f"layer_{i}"
            )(hh, mask=mask, is_training=is_training)

            if i == 7:
                hh_return = hh

        # 3. Final Norm and Output
        hh = transformer_core.layer_norm(hh)

        if mask is not None:
            hh *= mask[:, :, None]

        logits = transformer_core.conv1(
            hh, self._num_classes, init_scale=self._dense_init_scale)

        if return_embeddings:
            return logits, hh

        return logits


class Transformer(hk.Module):
  """Transformer tower."""

  def __init__(self,
               input_embedder,
               num_classes=1623,
               num_layers=8,
               num_heads=8,
               dropout_prob=0.1,
               self_att_init_scale=1.0,
               dense_init_scale=1.0,
               name=None):
    """Initialize the Transformer tower.

    Args:
      input_embedder: InputEmbedder object.
      num_classes: Total number of output classes.
      num_layers: Number of transformer blocks.
      num_heads: Number of transformer heads.
      dropout_prob: Dropout probability.
      self_att_init_scale: Scale for self-attention initialization.
      dense_init_scale: Scale for dense layer initialization.
      name: Optional name for the module.
    """
    super(Transformer, self).__init__(name=name)
    self._input_embedder = input_embedder
    self._num_classes = num_classes
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._self_att_init_scale = self_att_init_scale
    self._dense_init_scale = dense_init_scale

  # def __call__(self, examples, labels, mask=None, is_training=True):
  def __call__(self,
               examples,
               labels,
               mask=None,
               is_training=True,
               return_embeddings=False):
    """Call to the Transformer tower.

    Args:
      examples: input sequence of shape
        [batch_size, seq_len, height, width, channels]
      labels: input sequence of shape [batch_size, seq_len]
      mask: optional input mask of shape [batch_size, seq_len].
      is_training: if is currently training.

    Returns:
      outputs: output of the transformer tower
        of shape [batch_size, seq_len, channels].
    """
    # Encode the examples and labels.
    # jax.debug.print("label{}", labels)
    hh = self._input_embedder(examples, labels, is_training)
    hh = transformer_core.conv1(hh, 64, init_scale=self._dense_init_scale)

    if mask is not None:
      attention_mask = mask[:, None, None, :]
    else:
      attention_mask = None

    for i in range(self._num_layers):
      if mask is not None:
        hh *= mask[:, :, None]
      hh = transformer_core.TransformerBlock(
          causal=True,
          widening_factor=4,
          num_heads=self._num_heads,
          self_att_init_scale=self._self_att_init_scale,
          dense_init_scale=self._dense_init_scale,
          dropout_prob=self._dropout_prob)(
              hh, mask=attention_mask, is_training=is_training)
      if i == 7:
          hh_return = hh
    hh = transformer_core.layer_norm(hh)
    if mask is not None:
      hh *= mask[:, :, None]  # (B,S,E)
    # return transformer_core.conv1(
    #     hh, self._num_classes, init_scale=self._dense_init_scale)

    logits = transformer_core.conv1(
      hh, self._num_classes, init_scale=self._dense_init_scale)
    if return_embeddings:
      return logits, hh

    # 目标：往最后一个token中，数值最小的1500个维度的logits上，施加噪声
    # if is_training:
    #   rng = hk.next_rng_key()
    #
    #   batch_size = logits.shape[0]
    #   # 你的总类别数是 1623，目标是 1500
    #   target_k = 1500
    #
    #   # 1. 提取最后一个token的logits用于计算索引
    #   # shape: [Batch, Num_Classes]
    #   last_token_logits = logits[:, -1, :]
    #
    #   # 2. 找到数值最小的1500个索引
    #   # argsort 默认是从小到大排序，取前 target_k 个即为最小的 k 个
    #   # shape: [Batch, 1500]
    #   sorted_indices = jnp.argsort(last_token_logits, axis=-1)
    #   target_indices = sorted_indices[:, :target_k]
    #
    #   # 3. 生成噪声
    #   # shape: [Batch, 1500]
    #   noise = jax.random.normal(rng, shape=(batch_size, target_k)) * 3.0 + 5.0
    #
    #   # 4. 准备 Batch 索引以配合 JAX 的高级索引机制
    #   # shape: [Batch, 1] -> 广播成 [Batch, 1500]
    #   batch_indices = jnp.arange(batch_size)[:, None]
    #
    #   # 5. 将噪声添加到对应的索引位置
    #   # 语法解释: logits.at[Batch维度索引, 序列维度索引, Class维度索引].add(噪声)
    #   # 这里序列维度索引固定为 -1 (最后一个token)
    #   logits = logits.at[batch_indices, -1, target_indices].add(noise)

    return logits

class CoQE(hk.Module):
  """Transformer tower."""

  def __init__(self,
               input_embedder,
               num_classes=1623,
               num_layers=8,
               num_heads=8,
               dropout_prob=0.1,
               self_att_init_scale=1.0,
               dense_init_scale=1.0,
               name=None):
    """Initialize the Transformer tower.

    Args:
      input_embedder: InputEmbedder object.
      num_classes: Total number of output classes.
      num_layers: Number of transformer blocks.
      num_heads: Number of transformer heads.
      dropout_prob: Dropout probability.
      self_att_init_scale: Scale for self-attention initialization.
      dense_init_scale: Scale for dense layer initialization.
      name: Optional name for the module.
    """
    super(CoQE, self).__init__(name=name)
    self._input_embedder = input_embedder
    self._num_classes = num_classes
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_prob = dropout_prob
    self._self_att_init_scale = self_att_init_scale
    self._dense_init_scale = dense_init_scale
    self.head = hk.Linear(self._num_classes, with_bias=True, w_init=init.VarianceScaling(self._dense_init_scale))

  def __call__(self, examples, labels, train_step, mask=None, is_training=True):
    """Call to the Transformer tower.

    Args:
      examples: input sequence of shape
        [batch_size, seq_len, height, width, channels]
      labels: input sequence of shape [batch_size, seq_len]
      mask: optional input mask of shape [batch_size, seq_len].
      is_training: if is currently training.

    Returns:
      outputs: output of the transformer tower
        of shape [batch_size, seq_len, channels].
    """
    # Encode the examples and labels.

    hh = self._input_embedder(examples, labels, is_training)
    # 原来的写法
    # base_feat = hh

    # 参数更少的写法
    hh = transformer_core.conv1(hh, 64, init_scale=self._dense_init_scale)
    base_feat = hh

    if mask is not None:
      attention_mask = mask[:, None, None, :]
    else:
      attention_mask = None

    for _ in range(self._num_layers):
      if mask is not None:
        hh *= mask[:, :, None]
      hh = transformer_core.TransformerBlock(
          causal=True,
          widening_factor=4,
          num_heads=self._num_heads,
          self_att_init_scale=self._self_att_init_scale,
          dense_init_scale=self._dense_init_scale,
          dropout_prob=self._dropout_prob)(
              hh, mask=attention_mask, is_training=is_training)
    hh = transformer_core.layer_norm(hh)
    if mask is not None:
      hh *= mask[:, :, None]  # (B,S,E)

    # task_feat = jnp.roll(hh, shift=1, axis=1)
    task_feat = transformer_core.conv1(hh, base_feat.shape[-1], init_scale=self._dense_init_scale)
    D = task_feat.shape[-1]
    prototypes = hk.get_state("class_prototypes", shape=[self._num_classes, D],
                              init=init.VarianceScaling(self._dense_init_scale))

    label_task_feat = task_feat[:, 1::2, :]
    label_tokens = labels[:, :-1]

    def update_label_tokens(task_row, label_row):
      sums = jnp.zeros((self._num_classes, D))
      counts = jnp.zeros((self._num_classes, 1))

      def body(i, carry):
        s, c = carry
        label = label_row[i]
        vec = task_row[i]
        s = s.at[label].add(vec)
        c = c.at[label].add(1.0)
        return s, c

      return jax.lax.fori_loop(0, task_row.shape[0], body, (sums, counts))

    sum_batch, count_batch = jax.vmap(update_label_tokens)(label_task_feat, label_tokens)
    sum_total = jnp.sum(sum_batch, axis=0)        # [C, D]
    count_total = jnp.sum(count_batch, axis=0)    # [C, 1]
    new_values = sum_total / (count_total + 1e-6) # [C, D]

    # Step 5: update prototypes
    momentum = 0

    updated_prototypes = jnp.where(count_total > 0,
                                   momentum * prototypes + (1 - momentum) * new_values,
                                   prototypes)

    hk.set_state("class_prototypes", updated_prototypes)

    # Step 6: compute logits (dot product)
    context_logits = jnp.einsum("btd,cd->btc", base_feat, updated_prototypes)  # [B, S, C]

    base_logits = self.head(base_feat)

    # === NEW ===
    B, S, C = base_logits.shape
    last_t = S - 1
    last_logits = base_logits[:, last_t, :]  # [B, C]
    dim_noise = 1500
    _, min_idx = jax.lax.top_k(-last_logits, dim_noise)  # [B, 1000]

    # 生成 N(3, 5) 的噪声
    key = hk.next_rng_key()

    def get_noise(train_step, key, B, dim_noise):
        step = jnp.asarray(train_step, jnp.float32)

        # 计算区间编号（从0开始）
        k = jnp.maximum(0, jnp.floor((step - 1e4) / 1e4)).astype(jnp.int32)

        # 根据区间计算均值和标准差
        mu = 5.0 + k
        sigma = 3.0 + k
        # mu = 9.0 + k
        # sigma = 7.0 + k

        # 生成噪声
        noise = mu + sigma * jax.random.normal(key, (B, dim_noise))
        return noise

    noise = get_noise(train_step, key, B, dim_noise)
    # if train_step < 2e4:
    #     noise = 6.0 + 4.0 * jax.random.normal(key, (B, dim_noise))  # [B, 1000]
    # elif train_step < 6e4:
    #     noise = 8.0 + 6.0 * jax.random.normal(key, (B, dim_noise))
    # else:
    #     noise = 10.0 + 8.0 * jax.random.normal(key, (B, dim_noise))
    # 将噪声加到最小 1000 个 logit 上
    min_vals = jnp.take_along_axis(last_logits, min_idx, axis=-1)  # [B, 1000]
    new_vals = min_vals + noise
    # ablation
    # new_vals = min_vals
    # 写回（用 stop_gradient 切断梯度）
    last_logits_mod = last_logits.at[
        jnp.arange(B)[:, None], min_idx
    ].set(new_vals)
    last_logits_mod = jax.lax.stop_gradient(last_logits_mod)
    # 重新拼回 base_logits（仅用于后续计算）
    base_logits_mod = base_logits.at[:, last_t, :].set(last_logits_mod)
    # === END NEW ===

    def replace_last_step_logits(hh2, hh1, label_row):
        """
        Args:
            hh2: [seq_len, num_classes]
            hh1: [seq_len, num_classes]
            label_row: [seq_len]
        Returns:
            updated hh2
        """
        t = hh2.shape[0] - 1  # last timestep
        num_classes = hh2.shape[1]

        # Create one-hot mask of seen classes
        class_mask = jnp.any(jax.nn.one_hot(label_row, num_classes=num_classes), axis=0)  # [num_classes], bool

        # Get updated values and apply mask
        old_logits = hh2[t]
        new_logits = hh1[t]
        updated_logits = jnp.where(class_mask, new_logits + old_logits, old_logits)

        return hh2.at[t].set(updated_logits)

    if is_training:
        logits = jax.vmap(replace_last_step_logits)(base_logits_mod, context_logits, labels[:,:-1])
    else:
        logits = jax.vmap(replace_last_step_logits)(base_logits, context_logits, labels[:, :-1])

    return base_logits, logits