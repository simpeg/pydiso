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

The wrapper is written in cython and links MKL dynamically, through its single dynamic
library (`mkl_rt`). It needs to find the MKL header files and libraries to compile, which the
meson build backend does with `pkg-config` (the `mkl-sdl.pc` file MKL ships). Most
development installations of MKL provide this. For example, conda users can install the
necessary files with the `mkl-devel` package that is available on the default channel,
conda-forge channel, the intel channel, or others, e.g.

`conda install mkl-devel pkg-config`

If you have installed the `.pc` files to a non-standard location, set `PKG_CONFIG_PATH` to
point to that location.

After the necessary MKL files are accessible, you should be able to install by running

`pip install .`

in the installation directory.

### Building against MKL from PyPI instead of conda

Intel also publishes MKL to PyPI: `mkl` (runtime libraries, a regular dependency),
`mkl-devel` (import libs plus pkg-config files) and `mkl-include` (headers), the latter two
listed as build requirements alongside `pkgconf` (a `pkg-config` binary from PyPI). So a plain

`pip install .`

in a normal (non-conda) virtual environment pulls all of them in and builds against them
automatically, no `PKG_CONFIG_PATH` needed - `meson.build` locates `mkl-devel`'s installed
`.pc` files itself. conda-forge's `pydiso` package instead installs with
`pip install --no-deps` and supplies its own conda packages, as it already does for numpy
and scipy.

**Linux and Windows only.** Intel hasn't published MKL for macOS past 2023.2 to begin with,
and that release's PyPI wheel is also missing a symlink needed at link time, so the build
succeeds but the extension fails to import. conda-forge's macOS packaging doesn't have that
gap - use conda there instead.

## Prebuilt wheels

The prebuilt wheels on PyPI (see `.github/workflows/wheels.yml`) link the same way and depend
on the `mkl` package, which supplies the MKL libraries (and its threading runtime) at import
time. MKL is not bundled into them.

# Licensing

pydiso's own source code is MIT licensed (see `LICENSE`), and that is all any pydiso
distribution contains. MKL is a separate dependency, installed from its own package under
Intel's license (the Intel Simplified Software License on PyPI); pydiso does not redistribute it.
