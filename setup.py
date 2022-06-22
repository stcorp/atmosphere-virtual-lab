from setuptools import setup

setup(
    name="atmosphere-virtual-lab",
    version="0.1.11",
    author="S[&]T",
    url="https://github.com/stcorp/atmosphere-virtual-lab",
    description="Top-level package for an Atmosphere Virtual Lab installation",
    license="BSD",
    packages=["avl"],
    install_requires=["requests", "numpy", "panel", "plotly", "pyproj", "vtk", "ipyleaflet", "ipywidgets", "scipy", "matplotlib"],
    package_data={'': ['8k_earth_daymap.jpg']},
    python_requires='>=3.7',
)
