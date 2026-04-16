import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from tools.build_utils import HIPExtension

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()
DEFAULT_HIPCC = "/opt/rocm/bin/hipcc"

# -------- env switches --------
BUILD_TORCH = os.environ.get("GROUP_GEMM_PRIMUS_BUILD_TORCH", "1") == "1"

# -------- Supported GPU ARCHS --------
SUPPORTED_GPU_ARCHS = ["gfx942", "gfx950", "gfx90a"]

KERNEL_LIB_SONAME = "libgroup_gemm_primus_kernels.so"


class GroupGemmPrimusBuildExt(BuildExtension):
    KERNEL_EXT_NAME = "libgroup_gemm_primus_kernels"

    def get_ext_filename(self, ext_name: str) -> str:
        filename = super().get_ext_filename(ext_name)
        if ext_name == self.KERNEL_EXT_NAME:
            filename = os.path.join(*filename.split(os.sep)[:-1], KERNEL_LIB_SONAME)
        return filename

    def build_extension(self, ext):
        super().build_extension(ext)

        if ext.name == self.KERNEL_EXT_NAME:
            built_path = Path(self.get_ext_fullpath(ext.name))
            filename = built_path.name
            src_dst_dir = PROJECT_ROOT / "group_gemm_primus" / "lib"
            src_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, src_dst_dir / filename)
            build_dst_dir = Path(self.build_lib) / "group_gemm_primus" / "lib"
            build_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, build_dst_dir / filename)
            print(f"[GroupGemmPrimusBuildExt] Copied {filename} to:")
            print(f"  - {src_dst_dir}")
            print(f"  - {build_dst_dir}")


def all_files_in_dir(path, name_extensions=None):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            suffix = Path(name).suffix.lstrip(".")
            if name_extensions and suffix not in name_extensions:
                continue
            all_files.append(Path(dirname, name))
    return all_files


def setup_cxx_env():
    user_cxx = os.environ.get("CXX")
    if user_cxx:
        print(f"[groupGEMM-primus Setup] Using user-provided CXX: {user_cxx}")
    else:
        os.environ["CXX"] = DEFAULT_HIPCC
        print(f"[groupGEMM-primus Setup] No CXX provided. Defaulting to: {DEFAULT_HIPCC}")

    os.environ.setdefault("CMAKE_CXX_COMPILER", os.environ["CXX"])
    os.environ.setdefault("CMAKE_HIP_COMPILER", os.environ["CXX"])
    print(f"[groupGEMM-primus Setup] CMAKE_CXX_COMPILER set to: {os.environ['CMAKE_CXX_COMPILER']}")
    print(f"[groupGEMM-primus Setup] CMAKE_HIP_COMPILER set to: {os.environ['CMAKE_HIP_COMPILER']}")


def get_version():
    base_version = None
    with open(os.path.join("group_gemm_primus", "__init__.py")) as f:
        for line in f:
            match = re.match(r"^__version__\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                base_version = match.group(1)
                break
    if base_version is None:
        raise RuntimeError("Cannot find version.")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        return f"{base_version}+{commit}"  # PEP440
    except Exception:
        return base_version


def get_offload_archs():
    gpu_archs = os.environ.get("GPU_ARCHS", None)

    arch_list = []
    if gpu_archs is None or gpu_archs.strip() == "":
        import torch

        arch_list = [torch.cuda.get_device_properties(0).gcnArchName.split(":")[0].lower()]
    else:
        arch_list = [arch.strip().lower() for arch in gpu_archs.split(";")]

    assert len(arch_list) == 1, "groupGEMM-primus only supports single arch for now."

    macro_arch_list = []
    offload_arch_list = []
    for arch in arch_list:
        if arch in SUPPORTED_GPU_ARCHS:
            offload_arch_list.append(f"--offload-arch={arch}")
            macro_arch_list.append(f"-DPRIMUS_TURBO_{arch.upper()}")
        else:
            print(f"[WARNING] Ignoring unsupported GPU_ARCHS entry: {arch}")
    assert len(offload_arch_list) == 1, "groupGEMM-primus: expected exactly one --offload-arch."
    return offload_arch_list, macro_arch_list


def get_common_flags():
    arch = platform.machine().lower()
    extra_link_args = [
        "-Wl,-rpath,/opt/rocm/lib",
        f"-L/usr/lib/{arch}-linux-gnu",
        "-fgpu-rdc",
        "--hip-link",
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
        "-std=c++20",
    ]

    nvcc_flags = [
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_OPERATORS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT162_OPERATORS__",
        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        "-fno-offload-uniform-block",
        "-mllvm",
        "--lsr-drop-solution=1",
        "-mllvm",
        "-enable-post-misched=0",
        "-mllvm",
        "-amdgpu-coerce-illegal-types=1",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-std=c++20",
        "-fgpu-rdc",
        "-DDISABLE_ROCSHMEM",
    ]

    offload_arch_list, macro_arch_list = get_offload_archs()
    cxx_flags += macro_arch_list
    cxx_flags.append("-DDISABLE_ROCSHMEM")
    nvcc_flags += macro_arch_list
    nvcc_flags += offload_arch_list

    max_jobs = int(os.getenv("MAX_JOBS", "64"))
    nvcc_flags.append(f"-parallel-jobs={max_jobs}")

    return {
        "extra_link_args": extra_link_args,
        "extra_compile_args": {
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    }


def build_kernels_extension():
    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] += [
        "-shared",
        f"-Wl,-soname,{KERNEL_LIB_SONAME}",
    ]

    kernels_source_files = Path(PROJECT_ROOT / "csrc" / "kernels" / "grouped_gemm")
    kernels_sources = all_files_in_dir(kernels_source_files, name_extensions=["cpp", "cc", "cu"])

    include_dirs = [
        Path(PROJECT_ROOT / "csrc" / "include"),
        Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
        Path(PROJECT_ROOT / "csrc"),
    ]

    return HIPExtension(
        name="libgroup_gemm_primus_kernels",
        include_dirs=include_dirs,
        sources=kernels_sources,
        library_dirs=[],
        libraries=["hipblaslt"],
        **extra_flags,
    )


def build_torch_extension():
    if not BUILD_TORCH:
        return None

    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] = [
        "-Wl,-rpath,$ORIGIN/../lib",
        f"-L{PROJECT_ROOT / 'group_gemm_primus' / 'lib'}",
        "-lgroup_gemm_primus_kernels",
        *extra_flags.get("extra_link_args", []),
    ]

    pytorch_csrc_source_files = Path(PROJECT_ROOT / "csrc" / "pytorch")
    sources = all_files_in_dir(pytorch_csrc_source_files, name_extensions=["cpp", "cc", "cu"])

    return CUDAExtension(
        name="group_gemm_primus.pytorch._C",
        sources=sources,
        include_dirs=[
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
            Path(PROJECT_ROOT / "csrc"),
        ],
        **extra_flags,
    )


if __name__ == "__main__":
    setup_cxx_env()

    kernels_ext = build_kernels_extension()
    torch_ext = build_torch_extension()
    ext_modules = [kernels_ext] + ([torch_ext] if torch_ext is not None else [])

    setup(
        name="group-gemm-primus",
        version=get_version(),
        packages=find_packages(exclude=["tests", "tests.*"]),
        package_data={"group_gemm_primus": ["lib/*.so"]},
        ext_modules=ext_modules,
        cmdclass={"build_ext": GroupGemmPrimusBuildExt.with_options(use_ninja=True)},
        install_requires=[
            "hip-python",
        ],
    )
