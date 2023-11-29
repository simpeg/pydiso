# pydiso

Pydiso is a python wrapper for the pardiso solver. It is currently implemented for the
Intel MKL's version of pardiso. It's goal is to expose the full functionality of pardiso
to python, allowing the user to customize it as necessary for their use cases.

# Installation


### Installing from source

The wrapper is written in cython and links to the mkl libraries dynamically. Therefore,
it needs to find the necessary header files associated with the MKL installation to compile. You
will also need to clone this repository.

#### With Conda
For conda users, these headers can be installed with `mkl-devel` package that is available
on the default channel, conda-forge channel, the intel channel, or others, e.g.

`conda install mkl-devel`

#### Without Conda
If you are not using a conda environment, you can install the MKL header files with: 
`apt-get update && apt-get install intel-mkl`


#### Building and installing
If you installed with conda, it is likely that your respective numpy installations will then 
identify the location of the installed MKL. However if you run into issues automatically
finding the library headers, you will need to set the `MKLROOT` environment variable to
point to the correct location.

##### On Linux
The `mkl-rt` library and `mkl.h` may be in the same folder. In that case `MKLROOT` should point to 
that directory. If you installed mkl with `apt-get`, it is likely your directory structure will 
look something like this:

```
usr/
    include/
        mkl/
            mkl.h
            ...
    lib/
        x86_64-linux-gnu/
            libmkl_rt.so
            ...
```
In this case, set `MKLROOT` to the `usr/` directory (the directory containing `include` and `lib`), 
e.g. `export MKLROOT="/usr"`.

##### On Windows
The `mkl-rt.h` and `mkl-rt.lib` are not in the same folder but seperated a level into
`Library` and `Include` directories, and in this case `MKLROOT` would point to the folder
containing them.

##### On MacOS
The Intel MKL libraries are no longer supported on macOS. 

### Finishing installation
After the necessary MKL files are accessible, you should be able to do the standard install
script common to python packages. Navigate to the `pydiso` directory, then run either

`python setup.py install`

or, equivalently

`pip install .`.
