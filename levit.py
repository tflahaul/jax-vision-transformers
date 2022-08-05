from flax import linen as nn
from jax import numpy as jnp
from typing import Sequence

class Residual(nn.Module):
	func: nn.Module

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(inputs) + inputs

class FeedForward(nn.Module):
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		filters = inputs.shape[-1]
		out = nn.Conv(filters * 2, (1, 1), use_bias=False)(inputs)
		out = nn.BatchNorm(self.training, scale_init=nn.initializers.zeros, use_bias=False)(out)
		out = nn.hard_swish(out)
		out = nn.Conv(filters, (1, 1), use_bias=False)(out)
		out = nn.BatchNorm(self.training, scale_init=nn.initializers.zeros, use_bias=False)(out)
		return out

class LeViTAttention(nn.Module):
	dim: int
	num_heads: int
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, C = inputs.shape
		attn_b = self.param('bias', nn.initializers.normal(0.02), (1, 1, H * W, H * W))
		out = nn.Conv(self.num_heads * self.dim * 4, (1, 1), use_bias=False)(inputs)
		out = nn.BatchNorm(self.training, scale_init=nn.initializers.zeros, use_bias=False)(out)
		q, k, v1, v2 = (x.reshape(B, H * W, self.num_heads, self.dim).transpose(0, 2, 1, 3) for x in out.split(4, -1))
		v = jnp.concatenate((v1, v2), -1) # [HW, 2D]
		out = nn.softmax(jnp.matmul(q, jnp.moveaxis(k, 2, 3)) + attn_b)
		out = nn.hard_swish(jnp.matmul(out, v)).transpose((0, 2, 1, 3)).reshape(B, H, W, -1)
		out = nn.Conv(C, (1, 1), use_bias=False)(out)
		return out

class LeViT(nn.Module):
	out_features: int
	width: int
	filters: Sequence[int] = (32, 64, 128, 256)
	stage_heads: Sequence[int] = (4, 6, 8)
	stage_depths: Sequence[int] = (4, 4, 4)
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, _ = inputs.shape
		out = nn.Sequential([*(nn.Conv(C, (3, 3), strides=2) for C in self.filters)])(inputs)
		out = nn.Sequential([*(nn.Sequential([ # stage 1
				Residual(LeViTAttention(self.width, self.stage_heads[0], self.training)),
				Residual(FeedForward(self.training))]) for _ in range(self.stage_depths[0]))])(out)
		# shrink attention 8 heads [B, 7, 7, D]
		out = nn.Sequential([*(nn.Sequential([ # stage 2
				Residual(LeViTAttention(self.width, self.stage_heads[1], self.training)),
				Residual(FeedForward(self.training))]) for _ in range(self.stage_depths[1]))])(out)
		# shrink attention 12 heads [B, 4, 4, D]
		out = nn.Sequential([*(nn.Sequential([ # stage 3
				Residual(LeViTAttention(self.width, self.stage_heads[2], self.training)),
				Residual(FeedForward(self.training))]) for _ in range(self.stage_depths[2]))])(out)
		out = nn.avg_pool(out, window_shape=(14, 14)).reshape(B, -1) # window_shape=(4, 4)
		cls = nn.Dense(self.out_features)(out)
		dis = nn.Dense(self.out_features)(out)
		return cls, dis

import jax
key = jax.random.PRNGKey(0)
(out, _), params = LeViT(10, 16).init_with_output(key, jax.random.uniform(key, (1, 224, 224, 3)))
print(out.shape)
