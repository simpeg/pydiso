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

Intel also publishes MKL to PyPI, split across a few packages: `mkl` (runtime shared
libraries), `mkl-devel` (import libraries plus pkg-config/CMake files) and `mkl-include`
(headers). `mkl-devel` and `mkl-include` are listed in this project's build requirements,
and `mkl` is a regular runtime dependency, so a plain

`pip install .`

in a normal virtual environment (i.e. not conda) will pull all of them in automatically and
build against them, with no manual `PKG_CONFIG_PATH` configuration needed. This is only
relevant outside of conda; conda-forge's `pydiso` package installs with `pip install
--no-deps` and supplies its own `mkl`/`mkl-devel` conda packages instead, the same way it
already does for numpy and scipy.

Note that Intel has not published MKL for macOS on PyPI (or conda-forge) past version
2023.2 — pip will pick up whatever the newest available build is for your platform, which
means macOS is limited to that older release regardless.
