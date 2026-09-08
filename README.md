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
The meson build backend uses CMake's `find_package()` (via the `MKLConfig.cmake` file MKL
ships) to identify the locations of the mkl header files and library dynamic libraries. Most
development installations of MKL provide this. For example, conda users can install the
necessary configuration information with the `mkl-devel` package that is available on the
default channel, conda-forge channel, the intel channel, or others, e.g.

`conda install mkl-devel cmake`

If you have installed the configuration files to a non-standard location, you will need to set
`CMAKE_PREFIX_PATH` to point to that location.

After the necessary MKL files are accessible, you should be able to install by running

`pip install .`

in the installation directory.

### Building against MKL from PyPI instead of conda

Intel also publishes MKL to PyPI: `mkl` (runtime libraries, a regular dependency),
`mkl-devel` (import libs plus CMake config files) and `mkl-include` (headers), the latter
two listed as build requirements alongside `cmake` itself. So a plain

`pip install .`

in a normal (non-conda) virtual environment pulls all of them in and builds against them
automatically, no `CMAKE_PREFIX_PATH` needed - `meson.build` locates `mkl-devel`'s installed
`MKLConfig.cmake` itself. conda-forge's `pydiso` package instead installs with
`pip install --no-deps` and supplies its own conda packages, as it already does for numpy
and scipy.

**Linux and Windows only.** Intel hasn't published MKL for macOS past 2023.2 to begin with,
and that release's PyPI wheel is also missing a symlink CMake needs, so the build succeeds
but the extension fails to import. conda-forge's macOS packaging doesn't have that gap - use
conda there instead.

### Linking MKL statically

By default pydiso links MKL as a single dynamic library (`mkl_rt`). Two other options are
available via `-Dmkl-link=`:

- `dynamic`: link the specific interface/threading/core libraries needed instead of `mkl_rt`.
- `static`: link those same libraries statically, so they're compiled directly into the
  extension. With `-Dmkl-threading=seq` (the default), this produces a fully self-contained
  extension with no MKL runtime dependency at all - no `mkl` package needed at import time.
  With `iomp`/`tbb` threading, the threading runtime itself is still a separate shared
  library (`intel-openmp`/`tbb`), but MKL's own libraries are not.

Static linking needs the `mkl-static` package, which isn't installed automatically (it's an
827MB download only relevant to this option). Install it yourself first - `conda install
mkl-static` or `pip install mkl-static` - then build with `--no-build-isolation`:

`pip install --no-build-isolation --no-deps -Csetup-args=-Dmkl-link=static .`

The prebuilt wheels on PyPI (see `.github/workflows/wheels.yml`) are built this way with
`iomp` threading: `cibuildwheel`'s default repair step (`auditwheel`/`delvewheel`) bundles
the resulting `libiomp5` dependency into the wheel automatically, the same way it would
bundle e.g. OpenBLAS, so those wheels end up with no separate runtime MKL dependency either
while still solving with multiple threads.
