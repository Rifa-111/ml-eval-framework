"""Packaging for the experimental ML Eval Framework."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    description = fh.read()

setup(
    name="ml-eval-framework",
    version="0.1.0",
    author="Rifa Ferzana",
    description="Experimental model evaluation and data ablation toolkit",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rifa-111/ml-eval-framework",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "psutil>=5.9",
        "rich>=13.6",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1"],
        "torch": ["torch>=2.1"],
        "tracking": ["mlflow>=2.8", "wandb>=0.16"],
        "analysis": ["matplotlib>=3.8", "seaborn>=0.13", "pyyaml>=6.0"],
    },
    entry_points={"console_scripts": [
        "ml-benchmark=experiments.classification_benchmark:main",
        "ml-ablation=experiments.ablation_demo:main",
    ]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
