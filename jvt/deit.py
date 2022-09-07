import functools
import optax
import jax

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

class MHDPAttention(nn.Module):
	num_heads: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, L, E = inputs.shape
		out = nn.Dense(self.num_heads * E * 3, use_bias=True)(inputs)
		q, k, v = (x.reshape(B, L, self.num_heads, E) for x in out.split(3, -1))
		attn = nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, k, optimize=True) * (1 / jnp.sqrt(E)))
		out = jnp.einsum('bhwd,bdhv->bwhv', attn, v, optimize=True)
		out = nn.Dense(E, use_bias=False)(out.reshape(B, L, -1))
		return out

class DeiT(nn.Module):
	patch_size: int
	out_features: int
	width: int
	depth: int
	num_heads: int
	dim_ffn: int

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		P = self.patch_size
		out = nn.Conv(self.width, kernel_size=(P, P), strides=P, use_bias=False)(inputs)
		out = out.reshape(out.shape[0], -1, out.shape[3]) # [B, H, W, E] -> [B, L, E]
		pe = self.param('pe', nn.initializers.normal(0.02), (1, out.shape[1] + 2, self.width))
		ct = self.param('ct', nn.initializers.zeros, (1, 1, self.width)) # class token
		dt = self.param('dt', nn.initializers.zeros, (1, 1, self.width)) # distillation token
		out = jnp.concatenate((ct.repeat(out.shape[0], 0), out, dt.repeat(out.shape[0], 0)), axis=1)
		out = out + pe
		out = nn.Sequential([*(nn.Sequential([
				ResidualPreNorm(MHDPAttention(self.num_heads)),
				ResidualPreNorm(FeedForward(self.dim_ffn))
			]) for d in range(self.depth))])(out)
		cls = nn.Dense(self.dim_ffn)(out[:, 0])
		cls = nn.tanh(cls)
		cls = nn.Dense(self.out_features)(cls)
		dis = nn.Dense(self.out_features)(out[:, -1])
		return cls, dis



@functools.partial(jax.jit)
def kl_divergence(y: jnp.DeviceArray, y_hat: jnp.DeviceArray) -> jnp.DeviceArray:
	return jnp.sum(jnp.exp(y) * (y - y_hat), -1)

@functools.partial(jax.jit, static_argnames=('temp', 'alpha'))
def soft_distillation_loss(y: jnp.DeviceArray, y_s: jnp.DeviceArray, y_t: jnp.DeviceArray, temp: float, alpha: float) -> jnp.DeviceArray:
	div = kl_divergence(jax.nn.softmax(y_t / temp), jax.nn.softmax(y_s / temp)) * (temp ** 2)
	return ((1 - alpha) * optax.softmax_cross_entropy(y_s, y)) + (div * alpha)

@functools.partial(jax.jit)
def hard_distillation_loss(y: jnp.DeviceArray, y_s: jnp.DeviceArray, y_t: jnp.DeviceArray) -> jnp.DeviceArray:
	return (optax.softmax_cross_entropy(y_s, y) + optax.softmax_cross_entropy(y_s, y_t)) / 2
