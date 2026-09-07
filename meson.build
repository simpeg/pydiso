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
    # PyPI's mkl-devel drops its .pc files somewhere pkg-config doesn't
    # search by default. Ask mkl-devel's own installed package metadata
    # where they actually landed, rather than assuming a location relative
    # to sysconfig's "data" path: under pip build isolation, mkl-devel is
    # typically installed into a separate "overlay" layered onto the build
    # interpreter via PYTHONPATH, so sysconfig (which reflects that
    # interpreter's own prefix) points at the wrong, unrelated directory.
    # Point pkg-config at the real one as a default (a user-supplied
    # `-Dpkg_config_path=` still wins); harmless no-op when mkl-devel isn't
    # pip-installed at all, e.g. conda-forge, which puts its own mkl-devel
    # on PKG_CONFIG_PATH itself.
    'pkg_config_path=' + run_command('python',
      [
        '-c',
        '''
import os
import importlib.metadata as md

pkgconfig_dir = ""
try:
    dist = md.distribution("mkl-devel")
except md.PackageNotFoundError:
    dist = None

if dist is not None:
    for f in dist.files or ():
        if os.path.basename(str(f)) == "mkl-sdl.pc":
            candidate = os.path.dirname(os.path.normpath(str(dist.locate_file(f))))
            if os.path.isdir(candidate):
                pkgconfig_dir = candidate
            break

print(pkgconfig_dir)
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