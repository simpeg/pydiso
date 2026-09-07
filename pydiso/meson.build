cython_file = custom_target(
    input: '_mkl_solver.in.pyx',
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

# MKL-specific options.
#
# Located via MKLConfig.cmake (from mkl-devel) rather than pkg-config.
# CMake's find_package() always considers the prefix relative to the
# running `cmake` executable itself as a search root, and mkl-devel and
# cmake land in that same prefix regardless of source (a conda env, or the
# venv/overlay pip creates for build isolation) - so this is found with no
# extra help needed, unlike pkg-config's PKG_CONFIG_PATH, which depends on
# nothing else in the environment having already claimed it for an
# unrelated purpose (this happens in practice: actions/setup-python sets
# it for its own interpreter's pkgconfig files).
use_ilp64 = get_option('use-ilp64')
mkl_interface = use_ilp64 ? 'ilp64' : 'lp64'

_threading_opt = get_option('mkl-threading')
_threading_map = {
  'auto': 'sequential',
  'seq': 'sequential',
  'iomp': 'intel_thread',
  'gomp': 'gnu_thread',
  'tbb': 'tbb_thread',
}
mkl_threading = _threading_map[_threading_opt]

mkl_version_req = '>=2023.0'

use_sdl = get_option('use-sdl')

mkl_required = not get_option('dist_ignore_mkl_dep')

mkl_cmake_args = ['-DMKL_ARCH=intel64', '-DMKL_INTERFACE=' + mkl_interface]

if use_sdl
    mkl_dep = dependency('MKL',
        method: 'cmake',
        modules: ['MKL::MKL'],
        cmake_args: mkl_cmake_args + ['-DMKL_LINK=sdl'],
        version: mkl_version_req,
        required: mkl_required,
    )
else
    # find mkl
    mkl_dep = dependency('MKL',
        method: 'cmake',
        modules: ['MKL::MKL'],
        cmake_args: mkl_cmake_args + [
            '-DMKL_LINK=dynamic',
            '-DMKL_THREADING=' + mkl_threading,
        ],
        required: false,  # may be required, but we need to emit a custom error message
        version: mkl_version_req,
    )

    mkl_may_use_sdl = not use_ilp64 and _threading_opt in ['auto', 'iomp']

    # Insert a second try with MKL, because we may be rejecting older versions
    # or missing the requested dynamic variant. If so, we need to retry
    # with MKL SDL, and drop the version constraint.
    if not mkl_dep.found() and mkl_may_use_sdl
        mkl_dep = dependency('MKL',
            method: 'cmake',
            modules: ['MKL::MKL'],
            cmake_args: mkl_cmake_args + ['-DMKL_LINK=sdl'],
            required: mkl_required,
        )
    endif

endif

cython_args = []
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

c_undefined_ok = ['-Wno-maybe-uninitialized']
cython_c_args = [numpy_nodepr_api, '-DCYTHON_CCOMPLEX=0']

module_path = 'pydiso'

# PyPI's `mkl` wheel puts its shared libraries under `<prefix>/lib`, which
# isn't on the loader's default search path from a few directories down in
# site-packages. rpath both possible depths (venv vs. plain prefix install)
# so that's found without needing LD_LIBRARY_PATH/DYLD_LIBRARY_PATH; mirrors
# mkl-service. No-op where those dirs don't exist, e.g. conda-forge builds.
rpath_link_args = []
if host_machine.system() != 'windows'
    if host_machine.system() == 'darwin'
        origin = '@loader_path'
    else
        origin = '$ORIGIN'
    endif

    rpath_link_args = [
        '-Wl,-rpath,' + origin / '../..',
        '-Wl,-rpath,' + origin / '../../..',
    ]
endif

py.extension_module(
    '_mkl_solver',
    cython_file,
    cython_args: cython_args,
    c_args: cython_c_args,
    link_args: rpath_link_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep, mkl_dep],
)

python_sources = [
  '__init__.py', 'mkl_solver.py', '_init_helper.py'
]

py.install_sources(
  python_sources,
  subdir: module_path
)