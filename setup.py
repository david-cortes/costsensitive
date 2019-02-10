try:
  from setuptools import setup
except:
  from distutils.core import setup
import numpy as np
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = 'costsensitive',
  packages = ['costsensitive'],
  install_requires=[
   'numpy',
   'scipy',
   'joblib>=0.13',
   'cython'
],
  python_requires = ">=3",
  version = '0.1.2',
  description = 'Reductions for Cost-Sensitive Multi-Class Classification',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/costsensitive',
  keywords = ['cost sensitive multi class', 'cost-sensitive multi-class classification', 'weighted all pairs', 'filter tree'],
  classifiers = [],

  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension("costsensitive.vwrapper", sources=["costsensitive/vwrapper.pyx"], include_dirs=[np.get_include()], extra_link_args=["-fopenmp"], extra_compile_args=["-O2","-fopenmp"])]
)
