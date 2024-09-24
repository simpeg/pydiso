cython_file = custom_target(
    input: '_mkl_solver.pyx.in',
    output: '_mkl_solver.pyx',
    command: [py,
  '-c',
'''
import sys
from pathlib import Path
from Cython.Tempita import sub

template = Path(sys.argv[1]).read_text("utf8")
output = sub(template)
Path(sys.argv[2]).write_text(output, "utf8")
''', '@INPUT@', '@OUTPUT@']
)

# NumPy include directory
np_dep = dependency('numpy')
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']

# MKL-specific options
mkl_dep_name = 'mkl-dynamic'

use_ilp64 = get_option('use-ilp64')
if use_ilp64
  mkl_dep_name += '-ilp64'
else
  mkl_dep_name += '-lp64'
endif

# MKL-specific options
_threading_opt = get_option('mkl-threading')
if _threading_opt == 'auto'
  mkl_dep_name += '-seq'
else
  mkl_dep_name += '-' + _threading_opt
endif

mkl_version_req = '>=2023.0'

use_sdl = get_option('use-sdl')

if use_sdl
    mkl_dep = dependency('mkl-sdl', required: true)
else
    # find mkl
    mkl_dep = dependency(mkl_dep_name,
        required: false,  # may be required, but we need to emit a custom error message
        version: mkl_version_req,
    )

    mkl_may_use_sdl = not use_ilp64 and _threading_opt in ['auto', 'iomp']

    # Insert a second try with MKL, because we may be rejecting older versions
    # or missing it because no pkg-config installed. If so, we need to retry
    # with MKL SDL, and drop the version constraint.
    if not mkl_dep.found() and mkl_may_use_sdl
        mkl_dep = dependency('mkl-sdl', required: true)
    endif

endif

c_undefined_ok = ['-Wno-maybe-uninitialized']
cython_c_args = [numpy_nodepr_api, '-DCYTHON_CCOMPLEX=0']

module_path = 'pydiso'

py.extension_module(
    '_mkl_solver',
    cython_file,
    c_args: cython_c_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep, mkl_dep],
)

python_sources = [
  '__init__.py', 'mkl_solver.py'
]

py.install_sources(
  python_sources,
  subdir: module_path
)