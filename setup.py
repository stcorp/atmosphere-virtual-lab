from setuptools import setup

setup(
    name="atmosphere-virtual-lab",
    version="1.0",
    author="S[&]T",
    url="https://github.com/stcorp/atmosphere-virtual-lab",
    description="Top-level package for an Atmosphere Virtual Lab installation",
    license="BSD",
    packages=["avl"],
    install_requires=["requests"],
)
