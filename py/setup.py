import os
import sys
import glob
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from distutils.cmd import Command
from wheel.bdist_wheel import bdist_wheel

from torch.utils import cpp_extension
from shutil import copyfile, rmtree

import subprocess
import platform
import warnings

dir_path = os.path.dirname(os.path.realpath(__file__))

CXX11_ABI = False

JETPACK_VERSION = None

__version__ = '0.5.0a0'

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

if "--release" not in sys.argv:
    __version__ = __version__ + "+" + get_git_revision_short_hash()
else:
    sys.argv.remove("--release")


if "--use-cxx11-abi" in sys.argv:
    sys.argv.remove("--use-cxx11-abi")
    CXX11_ABI = True

if platform.uname().processor == "aarch64":
    if "--jetpack-version" in sys.argv:
        version_idx = sys.argv.index("--jetpack-version") + 1
        version = sys.argv[version_idx]
        sys.argv.remove(version)
        sys.argv.remove("--jetpack-version")
        if version == "4.5":
            JETPACK_VERSION = "4.5"
        elif version == "4.6":
            JETPACK_VERSION = "4.6"
    if not JETPACK_VERSION:
        warnings.warn("Assuming jetpack version to be 4.6, if not use the --jetpack-version option")
        JETPACK_VERSION = "4.6"


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


BAZEL_EXE = which("bazelisk")

if BAZEL_EXE is None:
    BAZEL_EXE = which("bazel")
    if BAZEL_EXE is None:
        sys.exit("Could not find bazel in PATH")


def build_libtrtorch_pre_cxx11_abi(develop=True, use_dist_dir=True, cxx11_abi=False):
    cmd = [BAZEL_EXE, "build"]
    cmd.append("//:libtrtorch")
    if develop:
        cmd.append("--compilation_mode=dbg")
    else:
        cmd.append("--compilation_mode=opt")
    if use_dist_dir:
        cmd.append("--distdir=third_party/dist_dir/x86_64-linux-gnu")
    if not cxx11_abi:
        cmd.append("--config=python")
    else:
        print("using CXX11 ABI build")

    if JETPACK_VERSION == "4.5":
        cmd.append("--platforms=//toolchains:jetpack_4.5")
        print("Jetpack version: 4.5")
    elif JETPACK_VERSION == "4.6":
        cmd.append("--platforms=//toolchains:jetpack_4.6")
        print("Jetpack version: 4.6")

    print("building libtrtorch")
    status_code = subprocess.run(cmd).returncode

    if status_code != 0:
        sys.exit(status_code)


def gen_version_file():
    if not os.path.exists(dir_path + '/trtorch/_version.py'):
        os.mknod(dir_path + '/trtorch/_version.py')

    with open(dir_path + '/trtorch/_version.py', 'w') as f:
        print("creating version file")
        f.write("__version__ = \"" + __version__ + '\"')


def copy_libtrtorch(multilinux=False):
    if not os.path.exists(dir_path + '/trtorch/lib'):
        os.makedirs(dir_path + '/trtorch/lib')

    print("copying library into module")
    if multilinux:
        copyfile(dir_path + "/build/libtrtorch_build/libtrtorch.so", dir_path + '/trtorch/lib/libtrtorch.so')
    else:
        os.system("tar -xzf ../bazel-bin/libtrtorch.tar.gz --strip-components=2 -C " + dir_path + "/trtorch")


class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        global CXX11_ABI
        build_libtrtorch_pre_cxx11_abi(develop=True, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtrtorch()
        develop.run(self)


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global CXX11_ABI
        build_libtrtorch_pre_cxx11_abi(develop=False, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtrtorch()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        global CXX11_ABI
        build_libtrtorch_pre_cxx11_abi(develop=False, cxx11_abi=CXX11_ABI)
        gen_version_file()
        copy_libtrtorch()
        bdist_wheel.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = [
        './build', './dist', './trtorch/__pycache__', './trtorch/lib', './trtorch/include', './trtorch/bin', './*.pyc',
        './*.tgz', './*.egg-info'
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print('Removing %s' % os.path.relpath(path))
                rmtree(path)


ext_modules = [
    cpp_extension.CUDAExtension(
        'trtorch._C', [
            'trtorch/csrc/trtorch_py.cpp',
            'trtorch/csrc/tensorrt_backend.cpp',
            'trtorch/csrc/tensorrt_classes.cpp',
            'trtorch/csrc/register_tensorrt_classes.cpp',
        ],
        library_dirs=[(dir_path + '/trtorch/lib/'), "/opt/conda/lib/python3.6/config-3.6m-x86_64-linux-gnu"],
        libraries=["trtorch"],
        include_dirs=[
            dir_path + "trtorch/csrc",
            dir_path + "/../",
            dir_path + "/../bazel-TRTorch/external/tensorrt/include",
        ],
        extra_compile_args=[
            "-Wno-deprecated",
            "-Wno-deprecated-declarations",
        ] + (["-D_GLIBCXX_USE_CXX11_ABI=1"] if CXX11_ABI else ["-D_GLIBCXX_USE_CXX11_ABI=0"]),
        extra_link_args=[
            "-Wno-deprecated", "-Wno-deprecated-declarations", "-Wl,--no-as-needed", "-ltrtorch",
            "-Wl,-rpath,$ORIGIN/lib", "-lpthread", "-ldl", "-lutil", "-lrt", "-lm", "-Xlinker", "-export-dynamic"
        ] + (["-D_GLIBCXX_USE_CXX11_ABI=1"] if CXX11_ABI else ["-D_GLIBCXX_USE_CXX11_ABI=0"]),
        undef_macros=["NDEBUG"])
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='trtorch',
      version=__version__,
      author='NVIDIA',
      author_email='narens@nvidia.com',
      url='https://nvidia.github.io/TRTorch',
      description='A compiler backend for PyTorch JIT targeting NVIDIA GPUs',
      long_description_content_type='text/markdown',
      long_description=long_description,
      ext_modules=ext_modules,
      install_requires=[
          'torch>=1.9.0<1.11.0',
      ],
      setup_requires=[],
      cmdclass={
          'install': InstallCommand,
          'clean': CleanCommand,
          'develop': DevelopCommand,
          'build_ext': cpp_extension.BuildExtension,
          'bdist_wheel': BdistCommand,
      },
      zip_safe=False,
      license="BSD",
      packages=find_packages(),
      classifiers=[
          "Development Status :: 4 - Beta", "Environment :: GPU :: NVIDIA CUDA",
          "License :: OSI Approved :: BSD License", "Intended Audience :: Developers",
          "Intended Audience :: Science/Research", "Operating System :: POSIX :: Linux", "Programming Language :: C++",
          "Programming Language :: Python", "Programming Language :: Python :: Implementation :: CPython",
          "Topic :: Scientific/Engineering", "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Software Development", "Topic :: Software Development :: Libraries"
      ],
      python_requires='>=3.6',
      include_package_data=True,
      package_data={
          'trtorch': [
              'lib/*', 'include/trtorch/*.h', 'include/trtorch/core/*.h', 'include/trtorch/core/conversion/*.h',
              'include/trtorch/core/conversion/conversionctx/*.h', 'include/trtorch/core/conversion/converters/*.h',
              'include/trtorch/core/conversion/evaluators/*.h', 'include/trtorch/core/conversion/tensorcontainer/*.h',
              'include/trtorch/core/conversion/var/*.h', 'include/trtorch/core/ir/*.h',
              'include/trtorch/core/lowering/*.h', 'include/trtorch/core/lowering/passes/*.h',
              'include/trtorch/core/partitioning/*.h', 'include/trtorch/core/plugins/*.h',
              'include/trtorch/core/plugins/impl/*.h', 'include/trtorch/core/runtime/*.h',
              'include/trtorch/core/util/*.h', 'include/trtorch/core/util/logging/*.h', 'bin/*', 'BUILD', 'WORKSPACE'
          ],
      },
      exclude_package_data={
          '': ['*.cpp'],
          'trtorch': ['csrc/*.cpp'],
      })
