# NumPy include directory
# The try-except is needed because when things are
# split across drives on Windows, there is no relative path and an exception
# gets raised. There may be other such cases, so add a catch-all and switch to
# an absolute path. Relative paths are needed when for example a virtualenv is
# placed inside the source tree; Meson rejects absolute paths to places inside
# the source tree.
# For cross-compilation it is often not possible to run the Python interpreter
# in order to retrieve numpy's include directory. It can be specified in the
# cross file instead:
#   [properties]
#   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
#
# This uses the path as is, and avoids running the interpreter.
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  incdir_numpy = run_command(py,
    [
      '-c',
      '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
  '''
    ],
    check: true
  ).stdout().strip()
else
  _incdir_numpy_abs = incdir_numpy
endif
inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

#MKL stuffs

use_ilp64 = get_option('use-ilp64')
if use_ilp64
  blas_interface = ['interface: ilp64']
else
  blas_interface = ['interface: lp64']
endif

# MKL-specific options
_threading_opt = get_option('mkl-threading')
if _threading_opt == 'auto'
  # Switch default to iomp once conda-forge missing openmp.pc issue is fixed
  mkl_opts = ['threading: seq']
else
  mkl_opts = ['threading: ' + _threading_opt]
endif
blas_opts = {'mkl': mkl_opts}
mkl_version_req = '>=2023.0'  # see gh-24824
mkl_may_use_sdl = not use_ilp64 and _threading_opt in ['auto', 'iomp']
# find mkl

mkl_dep = dependency('mkl',
    modules: ['cblas'] + blas_interface + mkl_opts,
    required: false,  # may be required, but we need to emit a custom error message
    version: mkl_version_req,
)
# Insert a second try with MKL, because we may be rejecting older versions
# or missing it because no pkg-config installed. If so, we need to retry
# with MKL SDL, and drop the version constraint (this always worked).
if not mkl_dep.found() and mkl_may_use_sdl
    mkl_dep = dependency('mkl', modules: ['cblas', 'sdl: true'], required: true)
endif

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
is_windows = host_machine.system() == 'windows'
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'
c_undefined_ok = ['-Wno-maybe-uninitialized']

cython_c_args = [numpy_nodepr_api, use_math_defines]

cy_line_trace = get_option('cy_line_trace')
if cy_line_trace
  cython_c_args += ['-DCYTHON_TRACE_NOGIL=1']
endif

cython_cpp_args = cython_c_args

module_path = 'pydiso'

py.extension_module(
    'mkl_solver',
    'mkl_solver.pyx',
    include_directories: incdir_numpy,
    c_args: cython_c_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep, mkl_dep],
)

python_sources = [
  '__init__.py',
]

py.install_sources(
  python_sources,
  subdir: module_path
)