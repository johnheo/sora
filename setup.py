from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sora",
    version="0.1.0",
    author="snoopy",
    description="Structure-informed sequence prediction with CATH protein structure data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnheo/sora",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "torchmetrics>=1.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyrootutils>=1.0.0",
        "rich>=12.0.0",
        "wandb>=0.15.0",
        "tensorboard>=2.10.0",
        "fair-esm>=2.0.0",
        "biopython>=1.79",
        "biotite>=0.36.0",
        "prody>=2.4.0",
        "mdtraj>=1.9.7",
        "nglview>=3.0.3",
        "plotly>=5.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "h5py>=3.7.0",
        "zarr>=2.12.0",
        "numba>=0.56.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",
            "dask-cuda>=22.8.0",
            "cudf-cu11>=22.8.0",
            "cuml-cu11>=22.8.0",
        ],
        "full": [
            "rdkit>=2022.9.1",
            "openmm>=7.7.0",
            "mpi4py>=3.1.0",
            "dask>=2022.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "byprot-train=train:main",
            "byprot-eval=eval:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
) 