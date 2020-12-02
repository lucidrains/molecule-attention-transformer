from setuptools import setup, find_packages

setup(
  name = 'molecule-attention-transformer',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'Molecule Attention Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/molecule-attention-transformer',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'molecules'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
