from flax import linen as nn
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

class ViT(nn.Module):
	patch_size: int
	out_features: int
	dim: int = 192
	depth: int = 12
	num_heads: int = 12
	dim_heads: int = 768
	dim_ff: int = 3072

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		P = self.patch_size
		out = nn.Conv(self.dim, kernel_size=(P, P), strides=P, use_bias=False)(inputs)
		out = out.reshape(out.shape[0], -1, out.shape[3]) # [B, H, W, E] -> [B, L, E]
		pe = self.param('pe', nn.initializers.uniform(), (1, *out.shape[-2:])) # position embedding
		out = out + pe
		out = nn.Sequential([
			*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.dim_heads, self.num_heads)),
				ResidualPreNorm(FeedForward(self.dim_ff))
			]) for d in range(self.depth))])(out)
		out = nn.Dense(self.out_features)(out)
		return out
