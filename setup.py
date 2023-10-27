from setuptools import setup

setup(
    name="atmosphere-virtual-lab",
    version="0.5",
    author="S[&]T",
    url="https://github.com/stcorp/atmosphere-virtual-lab",
    description="Top-level package for an Atmosphere Virtual Lab installation",
    license="BSD",
    packages=["avl"],
    install_requires=[
        "cmcrameri",
        "ipyleaflet",
        "ipywidgets",
        "matplotlib",
        "numpy",
        "panel",
        "plotly",
        "pyproj",
        "requests",
        "scipy",
        "vtk",
    ],
    package_data={"": ["8k_earth_daymap.jpg"]},
    python_requires=">=3.7",
)
