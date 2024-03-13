# pydiso

Pydiso is a python wrapper for the pardiso solver. It is currently implemented for the
Intel MKL's version of pardiso. Its goal is to expose the full functionality of pardiso
to python, allowing the user to customize it as necessary for their use cases.

# Installation


## Installing from source

The wrapper is written in cython and links to the mkl libraries dynamically. Therefore,
it needs to find the necessary header files associated with the MKL installation to compile.
The meson build backend uses pkg-config to identify the locations of the mkl header files
and library dynamic libraries. Most development installations of MKL should provide the
necessary pkg-config files for this. For example, conda users can be install the necessary
configuration information with `mkl-devel` package that is available on the default channel,
conda-forge channel, the intel channel, or others, e.g.

`conda install mkl-devel`

If you have installed the configuration files to a non-standard location, you will need to set
`PKG_CONFIG_PATH` to point to that location.


After the necessary MKL files are accessible, you should be able to install by running

`pip install .`

in the installation directory.
