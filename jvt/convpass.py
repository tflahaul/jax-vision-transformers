from flax import linen as nn
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

class ConvolutionalBypass(nn.Module):
	hidden_dim: int
	coef: float
	func: nn.Module

	def setup(self) -> None:
		self.hidden_conv = nn.Conv(self.hidden_dim, (3, 3), padding='same')

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, L, E = inputs.shape
		H, P = self.hidden_dim, int((L - 1) ** 0.5)
		out = nn.Dense(features=H)(inputs)
		out = nn.gelu(out)
		out = out.at[:, 0].set(self.hidden_conv(out[:, None, None, 0]).reshape(B, H))
		out = out.at[:, 1:].set(self.hidden_conv(out[:, 1:].reshape(B, P, P, H)).reshape(B, L - 1, H))
		out = nn.gelu(out)
		out = nn.Dense(features=E)(out)
		return (self.coef * out) + self.func(inputs)

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

class ConvPassViT(nn.Module):
	patch_size: int
	out_features: int
	width: int
	depth: int
	num_heads: int
	dim_ffn: int
	convp_dim: int
	convp_coef: float = 1.0

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
				ResidualPreNorm(ConvolutionalBypass(self.convp_dim, self.convp_coef, MHDPAttention(self.num_heads))),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(out)
		out = nn.Dense(self.dim_ffn)(out[:, 0]) # class token
		out = nn.tanh(out)
		out = nn.Dense(self.out_features)(out)
		return out
