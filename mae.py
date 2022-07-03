from flax import linen as nn
from jax import random as jrnd
from jax import numpy as jnp

class ResidualPreNorm(nn.Module):
	func: nn.Module

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(nn.LayerNorm(epsilon=1e-9)(inputs)) + inputs

class FeedForward(nn.Module):
	dim: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		out = nn.Dense(self.dim)(inputs)
		out = nn.gelu(out)
		out = nn.Dense(inputs.shape[-1])(out)
		return out

class MHDPAttention(nn.Module):
	dim: int
	num_heads: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, L, E = inputs.shape
		out = nn.Dense(self.num_heads * self.dim * 3, use_bias=False)(inputs)
		q, k, v = [x.reshape(B, L, self.num_heads, self.dim).transpose((0, 2, 1, 3)) for x in out.split(3, -1)]
		out = nn.softmax(jnp.matmul(q, jnp.moveaxis(k, 2, 3)) / jnp.sqrt(self.dim))
		out = jnp.matmul(out, v).transpose((0, 2, 1, 3)).reshape(B, L, -1)
		out = nn.Dense(E, use_bias=False)(out)
		return out

class MAE(nn.Module):
	key: jrnd.KeyArray
	patch_size: int
	masking_ratio: float
	dim: int
	depth: int
	num_heads: int
	dim_heads: int
	dim_ffn: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		P = self.patch_size
		out = nn.Conv(self.dim, (P, P), strides=P, use_bias=False)(inputs)
		out = out.reshape(out.shape[0], -1, out.shape[3]) # [B, H, W, E] -> [B, P, E]
		ct = self.param('ct', nn.initializers.xavier_uniform(), (1, 1, self.dim)) # class token
		mt = self.param('mt', nn.initializers.xavier_uniform(), (1, 1, self.dim)) # mask token
		pe = self.param('pe', nn.initializers.xavier_uniform(), (1, out.shape[1] + 1, self.dim))
		out = jnp.concatenate((ct.repeat(out.shape[0], 0), out), axis=1)
		out = out + pe

		mask = jrnd.permutation(self.key, jnp.arange(1, out.shape[1] + 1))
		patches = out[:, mask[:int(out.shape[1] * (1 - self.masking_ratio))]]

		patches = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.dim_heads, self.num_heads)),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(patches)

		out = mt.repeat(out.shape[0], 0).repeat(out.shape[1], 1)
		out = out.at[:, mask[:int(out.shape[1] * (1 - self.masking_ratio))]].set(patches)

		# decoder here

		return out




f = MAE(jrnd.PRNGKey(42),
	patch_size=16,
	masking_ratio=0.75,
	dim=64,
	depth=4,
	num_heads=3,
	dim_heads=16,
	dim_ffn=32)

f.init(jrnd.PRNGKey(0), jnp.ones((8, 224, 224, 3)))
