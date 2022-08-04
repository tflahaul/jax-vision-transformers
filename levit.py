from flax import linen as nn
from jax import numpy as jnp
from typing import Tuple

class Residual(nn.Module):
	func: nn.Module

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(inputs) + inputs

class FeedForward(nn.Module):
	dim: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		out = nn.Dense(self.dim)(inputs)
		out = nn.hard_swish(out)
		out = nn.Dense(inputs.shape[-1])(out)
		return out

class LeViTAttention(nn.Module):
	num_heads: int
	dim: int
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, C = inputs.shape
		attn_b = self.param('bias', nn.initializers.normal(0.02), (1, 1, H * W, H * W))
		out = nn.Conv(self.num_heads * self.dim * 3, (1, 1), use_bias=False)(inputs) # !
		out = nn.BatchNorm(self.training, scale_init=nn.initializers.zeros, use_bias=False)(out)
		q, k, v = (x.reshape(B, H * W, self.num_heads, self.dim).transpose(0, 2, 1, 3) for x in out.split(3, -1))
		out = nn.softmax(jnp.matmul(q, jnp.moveaxis(k, 2, 3)) + attn_b)
		out = nn.hard_swish(jnp.matmul(out, v)).transpose((0, 2, 1, 3)).reshape(B, H, W, -1)
		out = nn.Conv(C, (1, 1), use_bias=False)(out)
		return out

class LeViT(nn.Module):
	dim: int
	dim_ffn: int
	channels: Tuple[int] = (32, 64, 128, 256)
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		out = nn.Sequential([*(nn.Conv(C, (3, 3), strides=2) for C in self.channels)])(inputs)
		out = nn.Sequential([*(nn.Sequential([ # stage 1
				Residual(LeViTAttention(4, self.dim, self.training)),
				Residual(FeedForward(self.dim_ffn))
			]) for _ in range(4))])(out)
		# shrink attention 8 heads
		out = nn.Sequential([*(nn.Sequential([ # stage 2
				Residual(LeViTAttention(6, self.dim, self.training)),
				Residual(FeedForward(self.dim_ffn))
			]) for _ in range(4))])(out)
		# shrink attention 12 heads
		out = nn.Sequential([*(nn.Sequential([ # stage 3
				Residual(LeViTAttention(8, self.dim, self.training)),
				Residual(FeedForward(self.dim_ffn))
			]) for _ in range(4))])(out)
		# average pooling
		# cls / dist tokens
		return out

import jax
key = jax.random.PRNGKey(0)
out, params = LeViT(16, 32).init_with_output(key, jax.random.uniform(key, (1, 224, 224, 3)))
print(out)
