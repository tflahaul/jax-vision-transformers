from jax import random as jrnd
from jax import numpy as jnp
from flax import linen as nn
from typing import Sequence

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
	num_heads: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, L, E = inputs.shape
		out = nn.Dense(self.num_heads * E * 3, use_bias=False)(inputs)
		q, k, v = (x.reshape(B, L, self.num_heads, E) for x in out.split(3, -1))
		attn = nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, k, optimize=True) * (1 / jnp.sqrt(E)))
		out = jnp.einsum('bhwd,bdhv->bwhv', attn, v, optimize=True)
		out = nn.Dense(E, use_bias=False)(out.reshape(B, L, -1))
		return out

class MaskedEncoder(nn.Module):
	key: jrnd.KeyArray
	patch_size: int
	masking_ratio: float
	depth: int
	width: int
	num_heads: int
	dim_ffn: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		P = self.patch_size
		out = nn.Conv(self.width, (P, P), strides=P, use_bias=False)(inputs)
		out = out.reshape(out.shape[0], -1, out.shape[3]) # [B, H, W, E] -> [B, P, E]
		pe = self.param('pe', nn.initializers.normal(0.02), (1, out.shape[1] + 1, self.width))
		ct = self.param('ct', nn.initializers.zeros, (1, 1, self.width)) # class token
		out = jnp.concatenate((ct.repeat(out.shape[0], 0), out), axis=1)
		out = out + pe
		mask = jrnd.permutation(self.key, jnp.arange(1, out.shape[1] + 1))
		tokens = out[:, mask[:int(out.shape[1] * (1 - self.masking_ratio))]]
		tokens = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.num_heads)),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(tokens)
		return tokens, mask, out.shape

class MaskedDecoder(nn.Module):
	inputs_shape: Sequence[int]
	embeddings_shape: Sequence[int]
	masking_ratio: float
	patch_size: int
	depth: int
	width: int
	num_heads: int
	dim_ffn: int

	@nn.compact
	def __call__(self, patches: jnp.DeviceArray, mask: jnp.DeviceArray) -> jnp.DeviceArray:
		B, N, E = self.embeddings_shape
		mt = self.param('mt', nn.initializers.normal(0.02), (1, 1, E)) # mask token
		pe = self.param('pe', nn.initializers.normal(0.02), (1, N, E))
		out = mt.repeat(B, 0).repeat(N, 1)
		out = out.at[:, mask[:int(N * (1 - self.masking_ratio))]].set(patches)
		out = out + pe
		out = nn.Dense(self.width)(out)
		out = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.num_heads)),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(out)
		out = out[:, 1:] # remove extra class token
		out = nn.Dense((self.patch_size ** 2) * self.inputs_shape[-1])(out)
		out = out.reshape(self.inputs_shape)
		return out

class MAE(nn.Module):
	key: jrnd.KeyArray
	patch_size: int
	masking_ratio: float
	depth: int
	width: int
	num_heads: int
	dim_ffn: int
	dec_depth: int
	dec_width: int
	dec_heads: int
	dec_ffn_dim: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		tokens, mask, embeddings_shape = MaskedEncoder(
			self.key,
			self.patch_size,
			self.masking_ratio,
			self.depth,
			self.width,
			self.num_heads,
			self.dim_ffn)(inputs)
		out = MaskedDecoder(
			inputs.shape,
			embeddings_shape,
			self.masking_ratio,
			self.patch_size,
			self.dec_depth,
			self.dec_width,
			self.dec_heads,
			self.dec_ffn_dim)(tokens, mask)
		return out
