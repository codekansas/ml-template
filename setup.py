#!/usr/bin/env python
# pylint: disable=import-outside-toplevel
# mypy: ignore-errors

import functools
import multiprocessing
import os
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path
from typing import List

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def get_arch_list() -> str:
    import torch.cuda

    if torch.cuda.is_available():
        major_num, minor_num = torch.cuda.get_device_capability()
        return f"{major_num}.{minor_num}"
    arch_list: List[str] = []
    for arch in torch.cuda.get_arch_list():
        match = re.match(r"sm_(\d+)", arch)
        assert match, f"Invalid arch list: {torch.cuda.get_arch_list()}"
        arch_list.append(match.group(1))
    assert arch_list, f"Empty arch list: {torch.cuda.get_arch_list()} (did you install the wrong PyTorch version?)"
    return ";".join(".".join([i[:-1], i[-1:]]) for i in arch_list)


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])

        self.source_path = os.path.abspath(name)
        self.name = name


class CMakeBuild(build_ext):
    """Defines a build command for CMake projects."""

    cmake_prefix_path: str
    cmake_cxx_flags: str
    python_path: str

    def run(self) -> None:
        import pybind11  # pylint: disable=import-error
        import torch._C
        from torch.utils.cpp_extension import (
            CUDA_HOME,
            include_paths as torch_include_paths,
        )

        if not shutil.which("cmake"):
            raise RuntimeError("CMake installation not found")
        if CUDA_HOME is not None and not shutil.which("nvcc"):
            raise RuntimeError("NVCC installation not found")
        if torch.utils.cmake_prefix_path is None:
            raise RuntimeError("CMake prefix path not found")

        # Need to copy PyBind flags.
        cmake_cxx_flags: List[str] = []
        for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
            val = getattr(torch._C, f"_PYBIND11_{name}")
            if val is not None:
                cmake_cxx_flags += [f'-DPYBIND11_{name}=\\"{val}\\"']

        cmake_cxx_flags += [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]

        # Found this necessary for building on Apple M1 machine.
        cmake_cxx_flags += ["-fPIC", "-Wl,-undefined,dynamic_lookup", "-Wno-unused-command-line-argument"]

        # System include paths.
        cmake_include_dirs = [*torch_include_paths(cuda=CUDA_HOME is not None), pybind11.get_include()]
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")
        if python_include_path is not None:
            cmake_include_dirs += [python_include_path]
        cmake_cxx_flags += [f"-isystem {dir_name}" for dir_name in cmake_include_dirs]

        # Sets paths to various CMake stuff.
        self.cmake_prefix_path = ";".join([torch.utils.cmake_prefix_path, pybind11.get_cmake_dir()])
        self.cmake_cxx_flags = " ".join(cmake_cxx_flags)

        # Gets the path to the Python installation.
        if not (python_path := shutil.which("python")):
            raise RuntimeError("Python path not found")
        self.python_path = python_path

        for ext in self.extensions:
            assert isinstance(ext, CMakeExtension)
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        config = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(ext.name)}",
            f"-DCMAKE_PREFIX_PATH={self.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE:FILEPATH={self.python_path}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DCMAKE_CXX_FLAGS='{self.cmake_cxx_flags}'",
        ]

        env = os.environ.copy()

        # Additional CUDA arguments.
        if CUDA_HOME is not None:
            nvcc_path = shutil.which("nvcc")
            assert nvcc_path is not None
            cmake_args += [
                f"-DCUDA_TOOLKIT_ROOT_DIR='{CUDA_HOME}'",
                f"-DTORCH_CUDA_ARCH_LIST='{get_arch_list()}'",
                f"-DCMAKE_CUDA_COMPILER='{Path(nvcc_path).resolve()}'",
            ]

        # Builds CMake to a temp directory.
        build_temp = os.path.abspath(self.build_temp)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        subprocess.check_call(["cmake", f"-S{ext.source_path}", f"-B{build_temp}"] + cmake_args, env=env)

        # Compiles the project.
        build_lib = os.path.abspath(self.build_lib)
        if not os.path.exists(build_lib):
            os.makedirs(build_lib)
        subprocess.check_call(
            [
                "cmake",
                "--build",
                build_temp,
                "--",
                f"-j{multiprocessing.cpu_count()}",
            ],
            cwd=build_lib,
            env=env,
        )

        # Runs stubgen, if it is installed.
        if shutil.which("stubgen") is not None:
            project_root = Path(__file__).resolve().parent
            subprocess.check_call(["stubgen", "-p", "ml.cpp", "-o", "."], cwd=project_root, env=env)


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


@functools.lru_cache
def torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ModuleNotFoundError:
        return os.environ["TORCH_VERSION"]


@functools.lru_cache
def torchvision_version() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except ModuleNotFoundError:
        return os.environ["TORCHVISION_VERSION"]


def torch_version_str() -> str:
    return re.sub(r"[\.\-\+]", "", torch_version())


setup(
    name="ml",
    version="0.0.1",
    description="ML project template repository",
    author="Benjamin Bolte",
    url="https://github.com/codekansas/ml-template",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    setup_requires=[
        "mypy",  # For Stubgen
        "pybind11",
        f"torch=={torch_version()}",
    ],
    install_requires=[
        "ffmpeg-python",
        "matplotlib",
        "omegaconf",
        "opencv-python",
        "pandas",
        "tensorboard",
        "tqdm",
        f"torch=={torch_version()}",
        f"torchvision=={torchvision_version()}",
    ],
    ext_modules=[CMakeExtension("ml/cpp")],
    cmdclass={"build_ext": CMakeBuild},
    exclude_package_data={
        "ml": [
            "cpp/**/*.cpp",
        ],
    },
)
