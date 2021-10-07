import os
from os.path import join, abspath, dirname

base_path = abspath(dirname(__file__))


# Enable line tracing for coverage of cython files conditionally
ext_kwargs = {}
if os.environ.get("TEST_COV", None) is not None:
    ext_kwargs["define_macros"] = [("CYTHON_TRACE_NOGIL", 1)]


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    import numpy.distutils.system_info as sysinfo
    config = Configuration("pydiso", parent_package, top_path)

    try:
        from Cython.Build import cythonize
        cythonize(join(base_path, "mkl_solver.pyx"))
    except ImportError:
        pass

    # get information about mkl location
    mkl_root = os.environ.get('MKLROOT', None)
    if mkl_root is None:
        mkl_info = sysinfo.get_info('mkl')
    else:
        mkl_info = {
            'include_dirs': [join(mkl_root, 'include')],
            'library_dirs': [join(mkl_root, 'lib'), join(mkl_root, 'lib', 'intel64')],
            'libraries': ['mkl_rt']
        }

    mkl_include_dirs = mkl_info.get('include_dirs', [])
    mkl_library_dirs = mkl_info.get('library_dirs', [])
    mkl_libraries = mkl_info.get('libraries', ['mkl_rt'])

    config.add_extension(
        "mkl_solver",
        sources=["mkl_solver.c"],
        libraries=mkl_libraries,
        include_dirs=get_numpy_include_dirs() + mkl_include_dirs,
        library_dirs=mkl_library_dirs,
        extra_compile_args=['-w'],
        **ext_kwargs
    )

    return config
