try:
	from setuptools import setup
	from setuptools import Extension
except:
	from distutils.core import setup
	from distutils.extension import Extension
import numpy as np
import os, warnings
import subprocess
from sys import platform

found_omp = True
def set_omp_false():
	global found_omp
	found_omp = False

from Cython.Distutils import build_ext
## https://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_subclass( build_ext ):
	def build_extensions(self):
		if self.compiler.compiler_type == 'msvc':
			for e in self.extensions:
				e.extra_compile_args = ['/O2']
		else:
			self.add_march_native()
			self.add_openmp_linkage()

			for e in self.extensions:
				# e.extra_compile_args = ['-fopenmp', '-O2', '-march=native', '-std=c99']
				# e.extra_link_args = ['-fopenmp']
				e.extra_compile_args += ['-O2', '-std=c99']

		build_ext.build_extensions(self)

	def add_march_native(self):
		arg_march_native = "-march=native"
		arg_mcpu_native = "-mcpu=native"
		if self.test_supports_compile_arg(arg_march_native):
			for e in self.extensions:
				e.extra_compile_args.append(arg_march_native)
		elif self.test_supports_compile_arg(arg_mcpu_native):
			for e in self.extensions:
				e.extra_compile_args.append(arg_mcpu_native)

	def add_openmp_linkage(self):
		arg_omp1 = "-fopenmp"
		arg_omp2 = "-qopenmp"
		arg_omp3 = "-xopenmp"
		args_apple_omp = ["-Xclang", "-fopenmp", "-lomp"]
		if self.test_supports_compile_arg(arg_omp1):
			for e in self.extensions:
				e.extra_compile_args.append(arg_omp1)
				e.extra_link_args.append(arg_omp1)
		elif (platform[:3].lower() == "dar") and self.test_supports_compile_arg(args_apple_omp):
			for e in self.extensions:
				e.extra_compile_args += ["-Xclang", "-fopenmp"]
				e.extra_link_args += ["-lomp"]
		elif self.test_supports_compile_arg(arg_omp2):
			for e in self.extensions:
				e.extra_compile_args.append(arg_omp2)
				e.extra_link_args.append(arg_omp2)
		elif self.test_supports_compile_arg(arg_omp3):
			for e in self.extensions:
				e.extra_compile_args.append(arg_omp3)
				e.extra_link_args.append(arg_omp3)
		else:
			set_omp_false()


	def test_supports_compile_arg(self, comm):
		is_supported = False
		try:
			if not hasattr(self.compiler, "compiler"):
				return False
			if not isinstance(comm, list):
				comm = [comm]
			comm_str = " ".join(comm)
			fname = "costsensitive_compiler_testing.c"
			with open(fname, "w") as ftest:
				ftest.write(u"int main(int argc, char**argv) {return 0;}\n")
			cmd = list(self.compiler.compiler)
			val_good = subprocess.call(cmd + [fname])
			try:
				val = subprocess.call(cmd + comm + [fname])
				is_supported = (val == val_good)
			except:
				is_supported = False
		except:
			pass
		try:
			os.remove(fname)
		except:
			pass
		return is_supported

setup(
	name = 'costsensitive',
	packages = ['costsensitive'],
	install_requires=[
	 'numpy>=1.17',
	 'scipy',
	 'joblib>=0.13',
	 'cython'
	],
	python_requires = ">=3",
	version = '0.1.2.13-1',
	description = 'Reductions for Cost-Sensitive Multi-Class Classification',
	author = 'David Cortes',
	author_email = 'david.cortes.rivera@gmail.com',
	url = 'https://github.com/david-cortes/costsensitive',
	keywords = ['cost sensitive multi class', 'cost-sensitive multi-class classification', 'weighted all pairs', 'filter tree'],
	classifiers = [],

	cmdclass = {'build_ext': build_ext_subclass},
	ext_modules = [Extension("costsensitive._vwrapper", sources=["costsensitive/vwrapper.pyx"], include_dirs=[np.get_include()])]
)

if not found_omp:
	warnings.warn("\n\n\nCould not find OpenMP. Package will be built without multi-threading capabilities.")
