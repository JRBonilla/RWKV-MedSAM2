from setuptools import setup, find_packages

setup(
    name="rwkv_medsam2",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'mmengine',
        'hydra-core'
    ],
)