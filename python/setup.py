from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os

here = os.path.abspath('%s/..' % __file__)
objfiles = [os.path.abspath('../obj/%s' % f) for f in os.listdir('%s/../obj' % here) if 'yolo.o' not in f]

os.environ['CC'] = 'clang-3.6'

extensions = [
        Extension('yolo', ['yolo.pyx'], extra_objects=objfiles)
        ]

setup(
        name='YOLO',
        ext_modules = cythonize(extensions)
    )
