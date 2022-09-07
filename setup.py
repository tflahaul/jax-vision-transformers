from setuptools import setup, find_packages

setup(
	name='jvt',
	author='Thomas Flahault',
	description='Vision transformers with JAX & Flax',
	version='9.7.22',
	url='https://github.com/tflahaul/jax-vision-transformers',
	packages=find_packages(exclude=['examples']),
	install_requires=['jaxlib', 'jax', 'flax', 'optax'],
	keywords=['artificial intelligence', 'computer vision'],
	classifiers=[
		'Programming Language :: Python',
		'Topic :: Scientific/Engineering :: Artificial Intelligence'
	]
)
