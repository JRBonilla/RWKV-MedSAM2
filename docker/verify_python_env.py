#!/usr/bin/env python3
"""Verify the pinned training environment and prevent OpenCV/cuDNN conflicts."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


EXPECTED_DISTRIBUTIONS = {
    "cupy-cuda12x": "13.5.1",
    "eva-decord": "0.6.1",
    "gdown": "5.2.0",
    "h5py": "3.13.0",
    "huggingface-hub": "1.26.0",
    "hydra-core": "1.3.2",
    "imageio": "2.37.0",
    "imageio-ffmpeg": "0.6.0",
    "iopath": "0.1.10",
    "ipykernel": "6.30.0",
    "ipywidgets": "8.1.7",
    "jupyter": "1.1.1",
    "jupyterlab": "4.4.5",
    "matplotlib": "3.10.3",
    "moviepy": "1.0.3",
    "nibabel": "5.3.2",
    "ninja": "1.11.1.4",
    "notebook": "7.4.5",
    "numpy": "1.24.4",
    "openpyxl": "3.1.5",
    "pandas": "2.2.3",
    "pillow": "11.0.0",
    "psutil": "7.0.0",
    "pyarrow": "20.0.0",
    "pycocotools": "2.0.11",
    "pydicom": "2.4.4",
    "scikit-learn": "1.7.2",
    "scipy": "1.15.3",
    "SimpleITK": "2.5.0",
    "tciaclient": "0.0.3",
    "tensorboard": "2.20.0",
    "torch": "2.7.1+cu128",
    "torchaudio": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "tqdm": "4.67.1",
}
FORBIDDEN_OPENCV_WHEELS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def build_field(build_info: str, name: str) -> str:
    prefix = f"{name}:"
    for line in build_info.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    raise RuntimeError(f"OpenCV build information has no {name!r} field")


def verify_versions() -> None:
    mismatches = []
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            installed = "not installed"
        if installed != expected:
            mismatches.append(f"{distribution}: expected {expected}, found {installed}")
    if mismatches:
        raise RuntimeError("pinned package mismatch:\n  " + "\n  ".join(mismatches))


def verify_no_opencv_wheel() -> None:
    installed = []
    for distribution in FORBIDDEN_OPENCV_WHEELS:
        try:
            installed.append(f"{distribution}=={version(distribution)}")
        except PackageNotFoundError:
            pass
    if installed:
        raise RuntimeError(
            "custom CUDA OpenCV would be shadowed by Python wheels: " + ", ".join(installed)
        )

    unconditional = []
    for raw_requirement in requires("dripp") or []:
        requirement = Requirement(raw_requirement)
        if requirement.name.lower().replace("_", "-") != "opencv-python":
            continue
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            unconditional.append(raw_requirement)
    if unconditional:
        raise RuntimeError(f"DRIPP has an unconditional OpenCV wheel dependency: {unconditional}")


def verify_native_stack() -> None:
    cv2 = import_module("cv2")
    import_module("cupy")
    import_module("decord")
    import_module("huggingface_hub")
    import_module("nibabel")
    import_module("pandas")
    import_module("pycocotools")
    import_module("pydicom")
    import_module("SimpleITK")
    torch = import_module("torch")

    cv2_path = Path(cv2.__file__).resolve()
    if not cv2_path.is_relative_to(Path("/usr/local")):
        raise RuntimeError(f"cv2 did not load the /usr/local source build: {cv2_path}")
    if cv2.__version__ != "4.10.0":
        raise RuntimeError(f"expected OpenCV 4.10.0, found {cv2.__version__}")
    cudnn = build_field(cv2.getBuildInformation(), "cuDNN")
    if not cudnn.startswith("YES (ver 9.7.1"):
        raise RuntimeError(f"OpenCV is not using cuDNN 9.7.1: {cudnn}")

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"expected PyTorch CUDA 12.8, found {torch.version.cuda}")
    if torch.backends.cudnn.version() != 90701:
        raise RuntimeError(
            f"expected PyTorch cuDNN 9.7.1 (90701), found {torch.backends.cudnn.version()}"
        )


def verify_dependency_graph() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        details = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"pip dependency check failed:\n{details}")


def main() -> int:
    verify_versions()
    verify_no_opencv_wheel()
    verify_native_stack()
    verify_dependency_graph()
    print(
        "Python environment verification passed: pinned packages match, "
        "pip check passed, no OpenCV wheel is installed, and OpenCV/PyTorch "
        "share CUDA 12.8 with cuDNN 9.7.1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
