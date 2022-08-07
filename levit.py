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
		out = nn.Dense(inputs.shape[-1] * 2, use_bias=False)(inputs)
		out = nn.BatchNorm(self.training, bias_init=nn.initializers.zeros)(out)
		out = nn.hard_swish(out)
		out = nn.Dense(inputs.shape[-1], use_bias=False)(out)
		out = nn.BatchNorm(self.training, bias_init=nn.initializers.zeros)(out)
		return out

class LeViT_MHDPAttention(nn.Module):
	dim: int
	num_heads: int
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, C = inputs.shape
		D = self.dim * self.num_heads
		attn_b = self.param('bias', nn.initializers.zeros, (1, 1, H * W, H * W)) # !
		out = nn.Dense(D * (3 + 1), use_bias=False)(inputs)
		out = nn.BatchNorm(self.training, scale_init=nn.initializers.ones, use_bias=False)(out)
		q, k, v = (x.reshape(B, H * W, self.num_heads, -1).swapaxes(1, 2) for x in out.split((D, D * 2), -1))
		out = nn.softmax((jnp.matmul(q, k.swapaxes(2, 3)) + attn_b) / jnp.sqrt(self.dim))
		out = jnp.matmul(out, v).swapaxes(1, 2).reshape(B, H, W, -1)
		out = nn.Dense(C, use_bias=False)(nn.hard_swish(out))
		return out

class LeViT_SubsampleAttention(nn.Module):
	dim: int
	out_dim: int
	num_heads: int
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, _ = inputs.shape
		D = self.dim * self.num_heads
		out_hw = (((H - 1) // 2) + 1)
		attn_b = self.param('bias', nn.initializers.zeros, (1, 1, out_hw ** 2, H * W)) # !
		out = nn.Dense(D * (2 + 3), use_bias=False)(inputs)
		out = nn.BatchNorm(self.training, scale_init=nn.initializers.ones, use_bias=False)(out)
		k, v = (x.reshape(B, H * W, self.num_heads, -1).swapaxes(1, 2) for x in out.split((D,), -1))
		q = nn.avg_pool(inputs, window_shape=(1, 1), strides=(2, 2))
		q = nn.Dense(D, use_bias=False)(q)
		q = nn.BatchNorm(self.training, scale_init=nn.initializers.ones, use_bias=False)(q)
		q = q.reshape(B, -1, self.num_heads, self.dim).swapaxes(1, 2)
		out = nn.softmax((jnp.matmul(q, k.swapaxes(2, 3)) + attn_b) / jnp.sqrt(self.dim))
		out = jnp.matmul(out, v).swapaxes(1, 2).reshape(B, out_hw, out_hw, -1)
		out = nn.Dense(self.out_dim, use_bias=False)(nn.hard_swish(out))
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
		B, H, W, _ = inputs.shape # [B, 224, 224, 3]
		out = nn.Sequential([*(nn.Conv(C, (3, 3), strides=2) for C in self.filters)])(inputs)
		out = nn.Sequential([*(nn.Sequential([ # stage 1
				Residual(LeViT_MHDPAttention(self.width, self.stage_heads[0], self.training)),
				Residual(FeedForward(self.training))]) for _ in range(self.stage_depths[0]))])(out)
		out = LeViT_SubsampleAttention(self.width, 384, 8, training=self.training)(out) # [B, 7, 7, D]
		out = nn.Sequential([*(nn.Sequential([ # stage 2
				Residual(LeViT_MHDPAttention(self.width, self.stage_heads[1], self.training)),
				Residual(FeedForward(self.training))]) for _ in range(self.stage_depths[1]))])(out)
		out = LeViT_SubsampleAttention(self.width, 512, 12, training=self.training)(out) # [B, 4, 4, D]
		out = nn.Sequential([*(nn.Sequential([ # stage 3
				Residual(LeViT_MHDPAttention(self.width, self.stage_heads[2], self.training)),
				Residual(FeedForward(self.training))]) for _ in range(self.stage_depths[2]))])(out)
		out = nn.avg_pool(out, window_shape=(4, 4)).reshape(B, -1)
		cls = nn.Dense(self.out_features)(out)
		dis = nn.Dense(self.out_features)(out)
		return cls, dis
