# pydiso

Pydiso is a python wrapper for the pardiso solver. It is currently implemented for the
Intel MKL's version of pardiso. It's goal is to expose the full functionality of pardiso
to python, allowing the user to customize it as necessary for their use cases.

# Installation


### Installing from source

The wrapper is written in cython and links to the mkl libraries dynamically. Therefore,
it needs to find the necessary header files associated with the MKL installation to compile.
For conda users, these headers can be installed with `mkl-devel` package that is available
on the default channel, conda-forge channel, the intel channel, or others, e.g.

`conda install mkl-devel`

Most of the time, your respective numpy installations will then be intelligent enough to
identify the location of the installed MKL. However if you run into issues automatically
finding the library headers, you will need to set the `MKLROOT` environment variable to
point to the correct location. On Linux and MacOS the `mkl-rt` library and `mkl.h` are
usually in the same folder, and thus the root should point to that directory. On Windows
the `mkl-rt.h` and `mkl-rt.lib` are not in the same folder but seperated a level into
`Library` and `Include` directories, and in this case `MKLROOT` would point to the folder
containing them.

After the necessary MKL files are accessible, you should be able to do the standard install
script common to python packages by running either

`python setup.py install`

or, equivalently

`pip install .`

in the installation directory.
