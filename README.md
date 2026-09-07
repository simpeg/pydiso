# pydiso

Pydiso is a python wrapper for the pardiso solver. It is currently implemented for the
Intel MKL's version of pardiso. Its goal is to expose the full functionality of pardiso
to python, allowing the user to customize it as necessary for their use cases.

# Installation

## Installing with conda 

```
conda install pydiso --channel conda-forge
```


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

### Building against MKL from PyPI instead of conda

Intel also publishes MKL to PyPI: `mkl` (runtime libraries, a regular dependency),
`mkl-devel` (import libs plus pkg-config/CMake files) and `mkl-include` (headers), the
latter two listed as build requirements. So a plain

`pip install .`

in a normal (non-conda) virtual environment pulls all of them in and builds against them
automatically, no `PKG_CONFIG_PATH` needed. conda-forge's `pydiso` package instead installs
with `pip install --no-deps` and supplies its own conda packages, as it already does for
numpy and scipy.

Note: Intel hasn't published MKL for macOS past version 2023.2 (PyPI or conda-forge), so
that's what pip/conda will resolve there regardless of what's available elsewhere.
