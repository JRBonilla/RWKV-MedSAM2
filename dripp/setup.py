from setuptools import setup, find_packages

setup(
    name        = "dripp",
    version     = "0.1.0",
    description = "Dataset Regex Indexing & Preprocessing Pipeline",
    packages    = find_packages(where="."),
    package_data = {
        "dripp": ["defaults.ini"],
    },
    entry_points = {
        "console_scripts": [
            "dripp=dripp.cli:main",
        ],
    },
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
