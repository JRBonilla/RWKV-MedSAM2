"""
Define package metadata, dependencies, and the DRIPP command-line entry point.
"""

from setuptools import setup, find_packages

setup(
    name="dripp",
    version="0.1.0",
    description="Dataset Regex Indexing & Preprocessing Pipeline",
    packages=find_packages(where="."),
    package_data={
        "dripp": ["defaults.ini"],
    },
    entry_points={
        "console_scripts": [
            "dripp=dripp.cli:main",
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
        "SimpleITK",
        "tqdm",
        "scipy",
        "nibabel",
        "pydicom",
    ],
    extras_require={
        "opencv": ["opencv-python>=4.7,<5"],
        "gpu": ["cupy-cuda12x>=13"],
    },
    python_requires=">=3.7",
)
