from distutils.core import setup
from Cython.Build import cythonize


# python setup.py build_ext --inplace

setup(
    ext_modules = cythonize("epsilon_dgl_compiled.pyx")
)