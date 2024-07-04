from setuptools import setup, find_packages

setup(
    name="hazGAN",
    version="2.01",
    author="Alison Peard",
    author_email="alison.peard@gmail.com",
    description="GAN for storm footprint generation.",
    license="GNU v3",
    url="https://github.com/alisonpeard/hazGAN",

    packages=find_packages(),
    install_requires=[
        "scipy",
        "pandas",
        "numpy",
        "geopandas",
        "matplotlib"
    ]
)
