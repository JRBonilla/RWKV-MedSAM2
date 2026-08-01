#!/usr/bin/env python3
"""Fail a container build when the custom OpenCV installation is incomplete."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def build_field(build_info: str, name: str) -> str:
    prefix = f"{name}:"
    for line in build_info.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    raise RuntimeError(f"OpenCV build information has no {name!r} field")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the custom CUDA OpenCV build.")
    parser.add_argument("--opencv-version", default="4.10.0")
    parser.add_argument("--cuda-version-prefix", default="12.8")
    parser.add_argument("--cudnn-version-prefix", default="9.7.1")
    parser.add_argument(
        "--cudnn-library-root",
        type=Path,
        default=Path("/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"),
    )
    parser.add_argument("--cxx-standard", type=int, default=17)
    return parser.parse_args()


def verify_video_io() -> str:
    with tempfile.TemporaryDirectory(prefix="opencv-video-smoke-") as directory:
        path = str(Path(directory) / "smoke.avi")
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (16, 16),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open an MJPG VideoWriter")
        writer_backend = writer.getBackendName()
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.write(np.full((16, 16, 3), 255, dtype=np.uint8))
        writer.release()

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not reopen its video smoke-test output")
        capture_backend = capture.getBackendName()
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None or frame.shape != (16, 16, 3):
            raise RuntimeError("OpenCV FFmpeg video read/write smoke test failed")
        if writer_backend != "FFMPEG" or capture_backend != "FFMPEG":
            raise RuntimeError(
                f"expected FFmpeg video backend, got writer={writer_backend}, reader={capture_backend}"
            )
        return writer_backend


def verify_linkage(expected_cudnn_root: Path) -> Path:
    package_dir = Path(cv2.__file__).resolve().parent
    native_modules = sorted(package_dir.rglob("cv2*.so"))
    if not native_modules:
        raise RuntimeError(f"could not locate the native cv2 module under {package_dir}")
    result = subprocess.run(
        ["ldd", str(native_modules[0])],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    linkage = result.stdout + result.stderr
    if "not found" in linkage:
        missing = [line.strip() for line in linkage.splitlines() if "not found" in line]
        raise RuntimeError(f"cv2 has unresolved native libraries: {missing}")

    cudnn_lines = [line.strip() for line in linkage.splitlines() if "libcudnn.so" in line]
    if len(cudnn_lines) != 1 or "=>" not in cudnn_lines[0]:
        raise RuntimeError(f"expected one direct libcudnn dependency, found: {cudnn_lines}")
    resolved = Path(cudnn_lines[0].split("=>", 1)[1].split("(", 1)[0].strip()).resolve()
    expected_root = expected_cudnn_root.resolve()
    if not resolved.is_relative_to(expected_root):
        raise RuntimeError(
            f"cv2 resolves cuDNN from {resolved}, expected it under {expected_root}"
        )
    return native_modules[0]


def main() -> int:
    args = parse_args()
    build_info = cv2.getBuildInformation()

    if cv2.__version__ != args.opencv_version:
        raise RuntimeError(f"expected OpenCV {args.opencv_version}, found {cv2.__version__}")

    cxx_standard = build_field(build_info, "C++ standard")
    if not cxx_standard.startswith(str(args.cxx_standard)):
        raise RuntimeError(
            f"expected C++{args.cxx_standard}, OpenCV reports {cxx_standard}"
        )

    cuda = build_field(build_info, "NVIDIA CUDA")
    if not cuda.startswith(f"YES (ver {args.cuda_version_prefix}"):
        raise RuntimeError(f"expected CUDA {args.cuda_version_prefix}, OpenCV reports {cuda}")
    for feature in ("CUFFT", "CUBLAS", "FAST_MATH"):
        if feature not in cuda:
            raise RuntimeError(f"OpenCV CUDA build is missing {feature}: {cuda}")

    cudnn = build_field(build_info, "cuDNN")
    if not cudnn.startswith(f"YES (ver {args.cudnn_version_prefix}"):
        raise RuntimeError(
            f"expected cuDNN {args.cudnn_version_prefix}, OpenCV reports {cudnn}"
        )

    if build_field(build_info, "FFMPEG") != "YES":
        raise RuntimeError("OpenCV was not built with FFmpeg")

    modules = set(build_field(build_info, "To be built").split())
    required_modules = {
        "core",
        "cudaarithm",
        "cudaimgproc",
        "cudawarping",
        "dnn",
        "imgcodecs",
        "imgproc",
        "python3",
        "videoio",
    }
    missing_modules = sorted(required_modules - modules)
    if missing_modules:
        raise RuntimeError(f"OpenCV is missing required modules: {missing_modules}")

    if not hasattr(cv2, "cuda_GpuMat"):
        raise RuntimeError("Python bindings do not expose cv2.cuda_GpuMat")
    if not hasattr(cv2.dnn, "DNN_BACKEND_CUDA"):
        raise RuntimeError("Python bindings do not expose the CUDA DNN backend")

    sample = np.arange(64, dtype=np.uint8).reshape(8, 8)
    resized = cv2.resize(sample, (16, 16), interpolation=cv2.INTER_LINEAR)
    encoded, payload = cv2.imencode(".png", resized)
    decoded = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE) if encoded else None
    if decoded is None or decoded.shape != (16, 16):
        raise RuntimeError("OpenCV image processing/codec smoke test failed")

    video_backend = verify_video_io()
    native_module = verify_linkage(args.cudnn_library_root)
    print(
        "OpenCV verification passed: "
        f"version={cv2.__version__}, CUDA={cuda}, cuDNN={cudnn}, "
        f"C++={cxx_standard}, video={video_backend}, module={native_module}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
