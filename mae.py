from flax import linen as nn
from jax import random as jrnd
from jax import numpy as jnp

class ResidualPreNorm(nn.Module):
	func: nn.Module

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(nn.LayerNorm()(inputs)) + inputs

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
	depth: int
	width: int
	num_heads: int
	dim_heads: int
	dim_ffn: int
	dec_depth: int
	dec_width: int
	dec_heads: int
	dec_ffn_dim: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, C = inputs.shape
		P = self.patch_size
		out = nn.Conv(self.width, (P, P), strides=P, use_bias=False)(inputs)
		out = out.reshape(out.shape[0], -1, out.shape[3]) # [B, H, W, E] -> [B, P, E]
		ct = self.param('ct', nn.initializers.normal(0.02), (1, 1, self.width)) # class token
		epe = self.param('epe', nn.initializers.he_uniform(), (1, out.shape[1] + 1, self.width))
		out = jnp.concatenate((ct.repeat(out.shape[0], 0), out), axis=1)
		out = out + epe

		mask = jrnd.permutation(self.key, jnp.arange(1, out.shape[1] + 1))
		patches = out[:, mask[:int(out.shape[1] * (1 - self.masking_ratio))]]
		patches = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.dim_heads, self.num_heads)),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(patches)

		mt = self.param('mt', nn.initializers.normal(0.02), (1, 1, self.width)) # mask token
		dpe = self.param('dpe', nn.initializers.he_uniform(), (1, out.shape[1] + 1, self.width))
		out = mt.repeat(out.shape[0], 0).repeat(out.shape[1], 1)
		out = out.at[:, mask[:int(out.shape[1] * (1 - self.masking_ratio))]].set(patches)
		out = out + dpe

		out = nn.Dense(self.dec_width)(out)
		out = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.dec_width, self.dec_heads)),
				ResidualPreNorm(FeedForward(self.dec_ffn_dim))
			]) for d in range(self.depth))])(out)
		out = out[:, 1:] # remove extra class token
		out = nn.Dense(P * P * C)(out)
		out = out.reshape(B, H, W, C)
		return out
