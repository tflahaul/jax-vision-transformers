from flax import linen as nn
from jax import numpy as jnp

class ResidualPreNorm(nn.Module):
	func: nn.Module

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(nn.LayerNorm(1e-9)(inputs)) + inputs

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
	enable_dropout: bool
	dropout_rate: float

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, L, E = inputs.shape
		out = nn.Dense(self.num_heads * E * 3, use_bias=False)(inputs)
		q, k, v = (x.reshape(B, L, self.num_heads, E) for x in out.split(3, -1))
		attn = nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, k, optimize=True) * (E ** -0.5))
		attn = nn.Dropout(self.dropout_rate, (), (not self.enable_dropout))(attn)
		out = jnp.einsum('bhwd,bdhv->bwhv', attn, v, optimize=True)
		out = nn.Dense(E, use_bias=False)(out.reshape(B, L, -1))
		return out

class ViT(nn.Module):
	patch_size: int
	out_features: int
	width: int
	depth: int
	num_heads: int
	dim_ffn: int
	enable_dropout: bool = False
	dropout_rate: float = 0.1

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		P = self.patch_size
		out = nn.Conv(self.width, kernel_size=(P, P), strides=P, use_bias=False)(inputs)
		out = out.reshape(out.shape[0], -1, out.shape[3]) # [B, H, W, E] -> [B, L, E]
		pe = self.param('pe', nn.initializers.normal(0.02), (1, out.shape[1] + 1, self.width))
		ct = self.param('ct', nn.initializers.zeros, (1, 1, self.width))
		out = jnp.concatenate((ct.repeat(out.shape[0], axis=0), out), axis=1)
		out = out + pe
		out = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.num_heads, self.enable_dropout, self.dropout_rate)),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(out)
		out = nn.Dense(self.dim_ffn)(out[:, 0]) # class token
		out = nn.tanh(out)
		out = nn.Dense(self.out_features)(out)
		return out
