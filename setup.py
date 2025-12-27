#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opti-Oignon - Local LLM Optimization Suite
===========================================
Setup configuration for package installation.

Installation:
    pip install -e .              # Development mode (editable)
    pip install .                 # Production mode

After installation:
    opti-oignon --help            # Show all commands
    opti-oignon ui                # Launch web interface
    opti-oignon benchmark --help  # Benchmark help

Author: Léon
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Version
__version__ = "1.2.0"

setup(
    name="opti-oignon",
    version=__version__,
    author="Léon",
    author_email="",
    description="Local LLM optimization suite with intelligent routing, RAG, and multi-agent orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AntsAreRad/opti-oignon",
    license="MIT",
    
    # Package discovery
    packages=find_packages(),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Core dependencies
    install_requires=[
        "gradio>=4.0.0",
        "ollama>=0.2.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "pypdf>=3.0.0",
            "python-docx>=0.8.0",
            "openpyxl>=3.0.0",
        ],
        "all": [
            "pypdf>=3.0.0",
            "python-docx>=0.8.0",
            "openpyxl>=3.0.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    
    # CLI entry points - This is the key for global commands!
    entry_points={
        "console_scripts": [
            # Main unified CLI - handles all commands
            "opti-oignon=opti_oignon.main:main",
        ],
    },
    
    # Package data
    package_data={
        "opti_oignon": [
            "config/*.yaml",
            "config/*.yml",
            "prompts/*.yaml",
            "prompts/*.yml",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for searchability
    keywords=[
        "llm",
        "ollama",
        "local-ai",
        "routing",
        "rag",
        "multi-agent",
        "bioinformatics",
        "optimization",
        "cli",
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/AntsAreRad/opti-oignon/issues",
        "Source": "https://github.com/AntsAreRad/opti-oignon",
        "Documentation": "https://github.com/AntsAreRad/opti-oignon/tree/main/docs",
    },
)
