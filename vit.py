from flax import linen as nn
from jax import numpy as jnp

class Residual(nn.Module):
	func: nn.Module

	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(inputs) + inputs

class FeedForward(nn.Module):
	dim: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		out = nn.LayerNorm(epsilon=1e-9)(inputs)
		out = nn.Dense(self.dim)(out)
		out = nn.gelu(out)
		out = nn.Dense(inputs.shape[-1])(out)
		return out

class MHDPAttention(nn.Module):
	dim: int
	num_heads: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, L, E = inputs.shape
		out = nn.LayerNorm(epsilon=1e-9)(inputs)
		out = nn.Dense(self.num_heads * self.dim * 3, use_bias=False)(out)
		q, k, v = [x.reshape(B, L, self.num_heads, self.dim).transpose((0, 2, 1, 3)) for x in out.split(3, -1)]
		out = nn.softmax(jnp.matmul(q, jnp.moveaxis(k, 2, 3)) * (1 / jnp.sqrt(self.dim)))
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
		out = nn.Sequential([
			*(nn.Sequential([
				Residual(MHDPAttention(self.dim_heads, self.num_heads)),
				Residual(FeedForward(self.dim_ff))
			]) for d in range(self.depth))])(out)
		out = nn.Dense(self.out_features)(out)
		return out
