from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import numpy

here = os.path.abspath('%s/..' % __file__)
objfiles = [os.path.abspath('../obj/%s' % f) for f in os.listdir('%s/../obj' % here) if 'yolo.o' not in f]

os.environ['CC'] = 'clang'

extensions = [
        Extension('c_yolo', ['c_yolo.pyx'], extra_objects=objfiles, include_dirs=[numpy.get_include()])
        ]

setup(
        name='YOLO',
        ext_modules = cythonize(extensions)
    )
