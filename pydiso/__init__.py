__author__ = "SimPEG Team"
__license__ = "MIT"
__copyright__ = "2021, SimPEG Developers, http://simpeg.xyz"

from importlib.metadata import version, PackageNotFoundError

# Makes a pip-installed (as opposed to conda) MKL's DLLs findable on
# Windows; see _init_helper.py for details.
from . import _init_helper

del _init_helper

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
