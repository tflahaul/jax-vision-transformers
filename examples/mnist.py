# You need to have tensorflow and tensorflow-datasets installed to run this example

import tensorflow_datasets as tfds
import optax
import time
import jax

from flax.training import train_state, checkpoints
from functools import partial
from flax import linen as nn
from jax import numpy as jnp

from jvt import ViT

LEARNING_RATE = 2e-4
MAX_ITER = 8
BATCH_SIZE = 256
CKPT_DIR = 'checkpoints'

train_set = tfds.as_numpy(tfds.load('mnist', split='train', batch_size=BATCH_SIZE, as_supervised=True, data_dir='/tmp'))
test_set = tfds.as_numpy(tfds.load('mnist', split='test', batch_size=-1, as_supervised=True, data_dir='/tmp'))


@jax.jit
def apply_model(state, images, labels, key):
	def loss_fn(parameters):
		out = state.apply_fn(parameters, images, rngs={'dropout': key})
		loss = jnp.mean(optax.softmax_cross_entropy(out, jax.nn.one_hot(labels, 10)))
		return loss
	grad_fn = jax.value_and_grad(loss_fn)
	return grad_fn(state.params)


@partial(jax.jit, static_argnames='infer_fn')
def accuracy(parameters, infer_fn) -> float:
	images, labels = test_set
	return jnp.mean(jnp.argmax(infer_fn(parameters, images), -1) == labels)


def create_train_state(rng: jax.Array, f: nn.Module):
	parameters = jax.jit(f.init)(rng, jnp.ones((1, 28, 28, 1)))
	optimizer = optax.lion(LEARNING_RATE)
	return train_state.TrainState.create(
		apply_fn=jax.jit(f.apply),
		params=parameters,
		tx=optimizer)


def main() -> None:
	train_fn = ViT(28, 10, 48, 3, 6, 192, enable_dropout=True, dropout_rate=0.1)
	infer_fn = ViT(28, 10, 48, 3, 6, 192, enable_dropout=False).apply
	kp, kd = jax.random.split(jax.random.PRNGKey(seed=42))
	state = create_train_state({'params': kp, 'dropout': kd}, train_fn)
	print(f'Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))}')
	for epoch in range(1, MAX_ITER + 1):
		running_loss, start = 0, time.time()
		for images, labels in train_set:
			loss, gradients = apply_model(state, images, labels, kd)
			state = state.apply_gradients(grads=gradients)
			kd, _ = jax.random.split(kd)
			running_loss = running_loss + loss
		acc = accuracy(state.params, infer_fn)
		print(f'epoch {epoch:>2d}/{MAX_ITER}| loss={running_loss:.4f}, accuracy={acc:.3f}, time={time.time() - start:.2f}')
	print(f'Saved {checkpoints.save_checkpoint(CKPT_DIR, state, 0, overwrite=True)}')


if __name__ == '__main__':
	main()
