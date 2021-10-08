from distutils.core import setup
from setuptools import find_packages

import sys

def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("pydiso")

    return config

metadata = dict(
    name='pydiso',
    version='0.0.3',
    python_requires=">=3.6",
    setup_requires=[
        "numpy>=1.8",
        "cython>=0.2",
    ],
    install_requires=[
        'numpy>=1.8',
        'scipy>=0.13',
    ],
    author='SimPEG developers',
    author_email='josephrcapriotti@gmail.com',
    description="Wrapper for intel's pardiso implementation in the MKL",
    keywords='sparse, solver, wrapper',
    url='https://www.simpeg.xyz',
    download_url='https://github.com/jcapriot/pydiso-mkl',
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License'
)

if len(sys.argv) >= 2 and (
    "--help" in sys.argv[1:]
    or sys.argv[1] in ("--help-commands", "egg_info", "--version", "clean")
):
    # For these actions, NumPy is not required.
    #
    # They are required to succeed without Numpy, for example when
    # pip is used to install discretize when Numpy is not yet present in
    # the system.
    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup
else:
    if (len(sys.argv) >= 2 and sys.argv[1] in ("bdist_wheel", "bdist_egg")) or (
        "develop" in sys.argv
    ):
        # bdist_wheel/bdist_egg needs setuptools
        import setuptools

    from numpy.distutils.core import setup

    # Add the configuration to the setup dict when building
    # after numpy is installed
    metadata["configuration"] = configuration


setup(**metadata)
