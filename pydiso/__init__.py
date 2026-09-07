__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2021, SimPEG Developers, http://simpeg.xyz"

import os
import sys
from importlib.metadata import version, PackageNotFoundError

# A pip-installed (as opposed to conda) MKL puts its runtime DLLs under
# <env>/Library/bin, which conda's activation scripts add to PATH but a
# plain venv/virtualenv does not. Without this, importing the compiled
# `_mkl_solver` extension below would fail to locate `mkl_rt.*.dll` at
# import time. This mirrors the same fix in mkl-service's `_init_helper`.
if sys.platform == "win32":
    _dll_dir = os.path.join(sys.exec_prefix, "Library", "bin")
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)
    del _dll_dir

# Version
try:
    # - Released versions just tags:       0.8.0
    # - GitHub commits add .dev#+hash:     0.8.1.dev4+g2785721
    # - Uncommitted changes add timestamp: 0.8.1.dev4+g2785721.d20191022
    __version__ = version("pydiso")
except PackageNotFoundError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. discretize should be
    # installed properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")
