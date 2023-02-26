from setuptools import setup, Extension
import numpy as np

module = Extension('scan',
                   sources=['imageOptimizations.c'],
                   #extra_link_args=['-lgomp'],
                   extra_compile_args = ['-O3'],#'-fopenmp'],
                   include_dirs=[np.get_include()])

setup(name='scan',
      version='1.0',
      description='Example module',
      ext_modules=[module],
      install_requires=['numpy'])