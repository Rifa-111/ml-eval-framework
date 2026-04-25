"""
setup.py
========
Package setup for ml_eval_framework.
Install in development mode: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ml-eval-framework",
    version="1.0.0",
    author="ML Eval Framework",
    description="End-to-End Model Evaluation & Ablation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-benchmark=experiments.classification_benchmark:main",
            "ml-ablation=experiments.ablation_demo:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
