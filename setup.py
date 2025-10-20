from setuptools import setup, find_packages

setup(
    name="oulad-network-exploration",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "networkx",
        "seaborn",
        "jupyter"
    ],
)
