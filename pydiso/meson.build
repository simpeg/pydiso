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
# Located via MKLConfig.cmake (from mkl-devel) rather than pkg-config; see
# the top-level meson.build for how CMake is pointed at it.
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

mkl_link_opt = get_option('mkl-link')

mkl_required = not get_option('dist_ignore_mkl_dep')

mkl_cmake_args = ['-DMKL_ARCH=intel64', '-DMKL_INTERFACE=' + mkl_interface]

# Runtime DLL names for whichever variant gets linked; _init_helper.py
# uses these to find MKL's DLL directory on Windows. The interface layer
# (mkl_intel_lp64/ilp64) and mkl_gnu_thread aren't shipped as DLLs, only
# import libs, so they're left out.
mkl_sdl_library_names = ['mkl_rt']
mkl_dynamic_library_names = ['mkl_core', 'mkl_' + mkl_threading]

# A static build has no mkl_core.dll etc. (compiled directly into the
# extension), but OpenMP/TBB threading is still a separate shared library
# from a separate pip package (intel-openmp/tbb), if that's the choice.
_threading_runtime_library_names = {
  'sequential': [],
  'intel_thread': ['libiomp5md'],
  'gnu_thread': [],
  'tbb_thread': ['tbb12'],
}
mkl_static_library_names = _threading_runtime_library_names[mkl_threading]

if mkl_link_opt == 'sdl'
    mkl_dep = dependency('MKL',
        method: 'cmake',
        modules: ['MKL::MKL'],
        cmake_args: mkl_cmake_args + ['-DMKL_LINK=sdl'],
        version: mkl_version_req,
        required: mkl_required,
    )
    mkl_library_names = mkl_sdl_library_names
elif mkl_link_opt == 'static'
    # mkl-static (not installed automatically - see pyproject.toml).
    mkl_dep = dependency('MKL',
        method: 'cmake',
        modules: ['MKL::MKL'],
        cmake_args: mkl_cmake_args + [
            '-DMKL_LINK=static',
            '-DMKL_THREADING=' + mkl_threading,
        ],
        version: mkl_version_req,
        required: mkl_required,
    )
    mkl_library_names = mkl_static_library_names
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
    mkl_library_names = mkl_dynamic_library_names

    mkl_may_use_sdl = not use_ilp64 and _threading_opt in ['auto', 'iomp']

    # Retry with MKL SDL (dropping the version constraint) if the dynamic
    # variant wasn't found - may be an older version or a missing component.
    if not mkl_dep.found() and mkl_may_use_sdl
        mkl_dep = dependency('MKL',
            method: 'cmake',
            modules: ['MKL::MKL'],
            cmake_args: mkl_cmake_args + ['-DMKL_LINK=sdl'],
            required: mkl_required,
        )
        mkl_library_names = mkl_sdl_library_names
    endif

endif

_mkl_library_names_quoted = []
foreach _name : mkl_library_names
  _mkl_library_names_quoted += ['"' + _name + '"']
endforeach

if _mkl_library_names_quoted.length() == 0
  _mkl_library_names_repr = '()'
else
  _mkl_library_names_repr = '(' + ', '.join(_mkl_library_names_quoted) + ',)'
endif

_mkl_libs_config = configuration_data()
_mkl_libs_config.set('MKL_LIBRARY_NAMES_REPR', _mkl_library_names_repr)

mkl_libs_py = configure_file(
    input: '_mkl_libs.py.in',
    output: '_mkl_libs.py',
    configuration: _mkl_libs_config,
)

cython_args = []
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

c_undefined_ok = ['-Wno-maybe-uninitialized']
cython_c_args = [numpy_nodepr_api, '-DCYTHON_CCOMPLEX=0']

module_path = 'pydiso'

# PyPI's `mkl` wheel puts its libraries under `<prefix>/lib`, off the
# loader's default search path from site-packages. rpath both possible
# depths (venv vs. plain prefix) so it's found without
# LD_LIBRARY_PATH/DYLD_LIBRARY_PATH, mirroring mkl-service. No-op for
# conda-forge builds, where those dirs don't exist.
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

# Limited to the 3.12 API subset for one abi3 wheel per platform (see
# pyproject.toml's `limited-api`). Meson gates this on the
# `python.allow_limited_api` option, so `-Dpython.allow_limited_api=false`
# disables it cleanly for free-threaded builds, no conditional needed here.
py.extension_module(
    '_mkl_solver',
    cython_file,
    cython_args: cython_args,
    c_args: cython_c_args,
    link_args: rpath_link_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, np_dep, mkl_dep],
    limited_api: '3.12',
)

python_sources = [
  '__init__.py', 'mkl_solver.py', '_init_helper.py'
]

py.install_sources(
  python_sources + [mkl_libs_py],
  subdir: module_path
)