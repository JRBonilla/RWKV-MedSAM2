from setuptools import setup, find_packages

setup(
    name        = "dripp",
    version     = "0.1.0",
    description = "Dataset Regex Indexing & Preprocessing Pipeline",
    packages    = find_packages(where="."),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "SimpleITK",
        "tqdm",
        "SimpleITK",
        "scipy",
        "nibabel",
        "pydicom"
    ],
    python_requires= ">=3.7",
)