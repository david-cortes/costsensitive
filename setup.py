try:
	from setuptools import setup
	from setuptools import Extension
except:
	from distutils.core import setup
	from distutils.extension import Extension
import numpy as np
import os, warnings
from sys import platform


from Cython.Distutils import build_ext
## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
	def build_extensions(self):
		c = self.compiler.compiler_type
		if c == 'msvc': # visual studio
			for e in self.extensions:
				e.extra_compile_args = ['/O2']
		else: # gcc and clang
			for e in self.extensions:
				# e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c99']
				# e.extra_link_args = ['-fopenmp']
				e.extra_compile_args = ['-O2', '-march=native', '-std=c99']

			## Note: apple will by default alias 'gcc' to 'clang', and will ship its own "special"
			## 'clang' which has no OMP support and nowadays will purposefully fail to compile when passed
			## '-fopenmp' flags. If you are using mac, and have an OMP-capable compiler,
			## comment out the code below, and un-comment the lines above.
			if platform[:3] == "dar":
				apple_msg  = "\n\n\nMacOS detected. Package will be built without multi-threading capabilities, "
				apple_msg += "due to Apple's lack of OpenMP support in default Xcode installs. In order to enable it, "
				apple_msg += "install the package directly from GitHub: https://www.github.com/david-cortes/costsensitive\n"
				apple_msg += "And modify the setup.py file where this message is shown. "
				apple_msg += "You'll also need an OpenMP-capable compiler.\n\n\n"
				warnings.warn(apple_msg)
			else:
				for e in self.extensions:
					e.extra_compile_args.append('-fopenmp')
					e.extra_link_args.append('-fopenmp')
		build_ext.build_extensions(self)

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
	version = '0.1.2.10',
	description = 'Reductions for Cost-Sensitive Multi-Class Classification',
	author = 'David Cortes',
	author_email = 'david.cortes.rivera@gmail.com',
	url = 'https://github.com/david-cortes/costsensitive',
	keywords = ['cost sensitive multi class', 'cost-sensitive multi-class classification', 'weighted all pairs', 'filter tree'],
	classifiers = [],

	cmdclass = {'build_ext': build_ext_subclass},
	ext_modules = [Extension("costsensitive._vwrapper", sources=["costsensitive/vwrapper.pyx"], include_dirs=[np.get_include()])]
)
