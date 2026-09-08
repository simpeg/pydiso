# Adapted from mkl-service (https://github.com/IntelPython/mkl-service),
# mkl/_init_helper.py, under the BSD-3-Clause license below. Differences
# from upstream: this checks "not conda" instead of "is a real venv" (the
# latter misses a bare, non-venv Python, e.g. GitHub Actions'
# setup-python); the directory is found by trying, in order, MKLROOT (a
# manually installed oneAPI toolkit, if the user has set it - takes
# priority since it's an explicit, deliberate choice), the `mkl` package's
# own metadata, then a "<prefix>/Library/bin" guess as a last resort; and
# the library name(s) to look for come from _mkl_libs.py (build-time
# generated) rather than a hardcoded "mkl_rt", since a non-SDL build
# doesn't link that one at all.
#
# Copyright (c) 2025, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob
import os
import os.path
import sys

try:
    from ._mkl_libs import MKL_LIBRARY_NAMES
except ImportError:
    # Not generated, e.g. a non-meson/editable build predating this file.
    MKL_LIBRARY_NAMES = ("mkl_rt",)


def _has_any_dll(directory):
    return any(glob.glob(os.path.join(directory, name + "*.dll")) for name in MKL_LIBRARY_NAMES)


def _mklroot_dll_dir():
    # A manually installed oneAPI toolkit, e.g. via Intel's own installer
    # rather than pip/conda. Checked first: an explicit MKLROOT is a
    # deliberate choice that should win over guessing from installed
    # packages. "bin" is current oneAPI layout; "redist/intel64" was used
    # by older (pre-2024) toolkit releases.
    root = os.environ.get("MKLROOT")
    if not root:
        return None
    for candidate in (os.path.join(root, "bin"), os.path.join(root, "redist", "intel64")):
        if _has_any_dll(candidate):
            return candidate
    return None


def _mkl_package_dll_dir():
    import importlib.metadata as md

    try:
        dist = md.distribution("mkl")
    except md.PackageNotFoundError:
        return None

    for f in dist.files or ():
        base = os.path.basename(str(f)).lower()
        if any(base.startswith(name.lower()) for name in MKL_LIBRARY_NAMES):
            located = str(dist.locate_file(f))
            if os.path.isfile(located):
                return os.path.dirname(os.path.normpath(located))
            break
    return None


def _fallback_dll_dir():
    # Last resort if "mkl"'s metadata isn't discoverable; only used if a
    # DLL is actually there, so this can't add a bogus directory.
    candidate = os.path.join(sys.exec_prefix, "Library", "bin")
    return candidate if _has_any_dll(candidate) else None


def _add_mkl_dll_directory():
    is_conda = "CONDA_PREFIX" in os.environ or os.path.isdir(
        os.path.join(sys.prefix, "conda-meta")
    )
    if sys.platform != "win32" or is_conda:
        return

    for find_dll_dir in (_mklroot_dll_dir, _mkl_package_dll_dir, _fallback_dll_dir):
        dll_dir = find_dll_dir()
        if dll_dir is not None:
            os.add_dll_directory(dll_dir)
            return


_add_mkl_dll_directory()
del _add_mkl_dll_directory
