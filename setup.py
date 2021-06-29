
import setuptools
from numpy.distutils.core import setup
from Cython.Build import cythonize
from numpy.distutils.misc_util import Configuration

cythonize('pardiso_solver.pyx')

def configuration(parent_package='',top_path=None):
  config = Configuration('', parent_package, top_path)

  config.add_extension('pardiso_solver',
      sources=['pardiso_solver.c'],
                libraries=['mkl_rt'],
                include_dirs=['.'],
                extra_compile_args=['-w'],
                )
  return config

if __name__ == '__main__':
  setup(configuration=configuration)
