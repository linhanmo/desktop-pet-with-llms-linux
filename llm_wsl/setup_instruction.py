import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_sources():
    sources = glob.glob("csrc_instruction/*.cpp") + glob.glob("csrc_instruction/*.cu")
    return sources


def _get_ext_name():
    name = os.environ.get("LLM_WSL_EXTENSION_NAME", "").strip()
    return name if name else "llm_wsl_extension_instruction"


def _get_nvcc_args():
    default = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_86,code=sm_86",
    ]
    archs = os.environ.get("LLM_WSL_CUDA_ARCHS", "").strip()
    if not archs:
        return default
    items = []
    for part in archs.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = "".join(ch for ch in part if ch.isdigit())
        if not m:
            continue
        sm = int(m)
        items.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    if not items:
        return default
    extra = os.environ.get("LLM_WSL_NVCC_EXTRA", "").strip()
    extra_args = [x.strip() for x in extra.split() if x.strip()] if extra else []
    return ["-O3", "--use_fast_math", *items, *extra_args]


setup(
    name=_get_ext_name(),
    version="0.1.0",
    author="Captain-Mo",
    description="Custom CUDA kernels and C++ thread management for LlamaLite WSL (instruction tuning)",
    ext_modules=[
        CUDAExtension(
            name=_get_ext_name(),
            sources=get_sources(),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": _get_nvcc_args(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
