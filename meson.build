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
    # Before invoking CMake for real, meson guesses whether a config
    # package might exist by checking PATH entries stripped of a trailing
    # "bin"/"sbin" - missing Windows' pip/venv "Scripts" layout, so it
    # gives up before ever trying real CMake (which would succeed). Ask
    # mkl-devel's own metadata where MKLConfig.cmake actually is instead
    # of guessing; a user-supplied `-Dcmake_prefix_path=` still wins, and
    # this is a no-op when mkl-devel isn't pip-installed, e.g. conda-forge.
    'cmake_prefix_path=' + run_command('python',
      [
        '-c',
        '''
import os
import importlib.metadata as md

prefix = ""
try:
    dist = md.distribution("mkl-devel")
except md.PackageNotFoundError:
    dist = None

if dist is not None:
    for f in dist.files or ():
        if os.path.basename(str(f)) == "MKLConfig.cmake":
            # .../<prefix>/lib/cmake/mkl/MKLConfig.cmake
            candidate = os.path.normpath(str(dist.locate_file(f)))
            for _ in range(4):
                candidate = os.path.dirname(candidate)
            if os.path.isdir(candidate):
                prefix = candidate
            break

print(prefix)
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