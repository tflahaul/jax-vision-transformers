from flax import linen as nn
from jax import numpy as jnp
from typing import Sequence

class Residual(nn.Module):
	func: nn.Module

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		return self.func(inputs) + inputs

class FeedForward(nn.Module):
	scale_factor: int = 2
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		out = nn.Dense(inputs.shape[-1] * self.scale_factor, use_bias=False)(inputs)
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
		out = nn.softmax((jnp.matmul(q, k.swapaxes(2, 3)) / jnp.sqrt(self.dim)) + attn_b)
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
		out = nn.softmax((jnp.matmul(q, k.swapaxes(2, 3)) / jnp.sqrt(self.dim)) + attn_b)
		out = jnp.matmul(out, v).swapaxes(1, 2).reshape(B, out_hw, out_hw, -1)
		out = nn.Dense(self.out_dim, use_bias=False)(nn.hard_swish(out))
		return out

class LeViT(nn.Module):
	out_features: int
	width: int
	filters: Sequence[int]
	subattn_out_dim: Sequence[int]
	stage_depths: Sequence[int]
	stage_heads: Sequence[int]
	subattn_heads: Sequence[int]
	ffn_scale_factor: int = 2
	training: bool = False

	@nn.compact
	def __call__(self, inputs: jnp.DeviceArray) -> jnp.DeviceArray:
		B, H, W, C = inputs.shape # [B, 224, 224, 3]
		out = nn.Sequential([*(nn.Conv(C, (3, 3), strides=2) for C in self.filters)])(inputs)
		out = nn.Sequential([*(nn.Sequential([ # stage 1
				Residual(LeViT_MHDPAttention(self.width, self.stage_heads[0], self.training)),
				Residual(FeedForward(self.ffn_scale_factor, self.training))
			]) for _ in range(self.stage_depths[0]))])(out)
		out = LeViT_SubsampleAttention(self.width, self.subattn_out_dim[0], self.subattn_heads[0], self.training)(out)
		out = nn.Sequential([*(nn.Sequential([ # stage 2
				Residual(LeViT_MHDPAttention(self.width, self.stage_heads[1], self.training)),
				Residual(FeedForward(self.ffn_scale_factor, self.training))
			]) for _ in range(self.stage_depths[1]))])(out)
		out = LeViT_SubsampleAttention(self.width, self.subattn_out_dim[1], self.subattn_heads[1], self.training)(out)
		out = nn.Sequential([*(nn.Sequential([ # stage 3
				Residual(LeViT_MHDPAttention(self.width, self.stage_heads[2], self.training)),
				Residual(FeedForward(self.ffn_scale_factor, self.training))
			]) for _ in range(self.stage_depths[2]))])(out)
		out = nn.avg_pool(out, window_shape=(4, 4)).reshape(B, -1)
		cls = nn.Dense(self.out_features)(out)
		dis = nn.Dense(self.out_features)(out)
		return cls, dis




def LeViT_128S(out_features: int, training: bool = False) -> nn.Module:
	return LeViT(
		out_features=out_features,
		width=16,
		filters=(16, 32, 64, 128),
		subattn_out_dim=(256, 384),
		stage_depths=(2, 3, 4),
		stage_heads=(4, 6, 8),
		subattn_heads=(8, 16),
		training=training)

def LeViT_128(out_features: int, training: bool = False) -> nn.Module:
	return LeViT(
		out_features=out_features,
		width=16,
		filters=(16, 32, 64, 128),
		subattn_out_dim=(256, 384),
		stage_depths=(4, 4, 4),
		stage_heads=(4, 8, 12),
		subattn_heads=(8, 16),
		training=training)

def LeViT_192(out_features: int, training: bool = False) -> nn.Module:
	return LeViT(
		out_features=out_features,
		width=32,
		filters=(32, 64, 128, 192),
		subattn_out_dim=(288, 384),
		stage_depths=(4, 4, 4),
		stage_heads=(3, 5, 6),
		subattn_heads=(6, 9),
		training=training)

def LeViT_256(out_features: int, training: bool = False) -> nn.Module:
	return LeViT(
		out_features=out_features,
		width=32,
		filters=(32, 64, 128, 256),
		subattn_out_dim=(384, 512),
		stage_depths=(4, 4, 4),
		stage_heads=(4, 6, 8),
		subattn_heads=(8, 12),
		training=training)

def LeViT_384(out_features: int, training: bool = False) -> nn.Module:
	return LeViT(
		out_features=out_features,
		width=32,
		filters=(64, 128, 256, 384),
		subattn_out_dim=(512, 768),
		stage_depths=(4, 4, 4),
		stage_heads=(6, 9, 12),
		subattn_heads=(12, 18),
		training=training)
