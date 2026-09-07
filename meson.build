project(
  'pydiso',
  'c', 'cython',
  # Note that the git commit hash cannot be added dynamically here
  # (it is dynamically generated though setuptools_scm)
  version: run_command('python',
    [
      '-c',
      '''
from setuptools_scm import get_version
print(get_version())'''
    ],
    check: true
).stdout().strip(),

  license: 'MIT',
  meson_version: '>= 1.4.0',
  default_options: [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    # PyPI's mkl-devel/mkl-include drop their .pc files into this
    # interpreter's sysconfig "data" prefix, which pkg-config doesn't search
    # by default. Point it there as a default (a user-supplied
    # `-Dpkg_config_path=` still wins); harmless no-op when that dir doesn't
    # exist, e.g. conda-forge, which puts mkl-devel on PKG_CONFIG_PATH itself.
    'pkg_config_path=' + run_command('python',
      [
        '-c',
        '''
import os
import sysconfig

data_dir = sysconfig.get_path("data")
candidate = (
    os.path.join(data_dir, "Library", "lib", "pkgconfig")
    if os.name == "nt"
    else os.path.join(data_dir, "lib", "pkgconfig")
)
print(candidate if os.path.isdir(candidate) else "")
'''
      ],
      check: true
    ).stdout().strip(),
  ],
)

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')
# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])

_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
)
add_project_arguments(_global_c_args, language : 'c')

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for SciPy). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

subdir('pydiso')