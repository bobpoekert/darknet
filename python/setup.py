from distutils.core import setup
from Cython.Build import cythonize
import os

os.environ['CC'] = 'clang-3.6'

setup(
        name='YOLO',
        ext_modules = cythonize('yolo.pyx')
    )
