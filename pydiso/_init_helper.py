# Adapted from mkl-service (https://github.com/IntelPython/mkl-service),
# mkl/_init_helper.py, under the BSD-3-Clause license below. Differences
# from upstream: this checks "not conda" instead of "is a real venv" (the
# latter misses a bare, non-venv Python, e.g. GitHub Actions'
# setup-python); the DLL directory is found via mkl's own package
# metadata rather than assumed to be "<prefix>/Library/bin"; and the
# library name(s) to look for come from _mkl_libs.py (build-time
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


def _add_mkl_dll_directory():
    is_conda = "CONDA_PREFIX" in os.environ or os.path.isdir(
        os.path.join(sys.prefix, "conda-meta")
    )
    if sys.platform != "win32" or is_conda:
        return

    import importlib.metadata as md

    try:
        dist = md.distribution("mkl")
    except md.PackageNotFoundError:
        dist = None

    dll_dir = None
    if dist is not None:
        for f in dist.files or ():
            base = os.path.basename(str(f)).lower()
            if any(base.startswith(name.lower()) for name in MKL_LIBRARY_NAMES):
                located = str(dist.locate_file(f))
                if os.path.isfile(located):
                    dll_dir = os.path.dirname(os.path.normpath(located))
                break

    if dll_dir is None:
        # Fallback if "mkl"'s metadata isn't discoverable; only used if a
        # DLL is actually there, so this can't add a bogus directory.
        fallback = os.path.join(sys.exec_prefix, "Library", "bin")
        if any(glob.glob(os.path.join(fallback, name + "*.dll")) for name in MKL_LIBRARY_NAMES):
            dll_dir = fallback

    if dll_dir is not None:
        os.add_dll_directory(dll_dir)


_add_mkl_dll_directory()
del _add_mkl_dll_directory
