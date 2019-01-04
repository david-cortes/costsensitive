from distutils.core import setup
setup(
  name = 'costsensitive',
  packages = ['costsensitive'],
  install_requires=[
   'pandas',
   'numpy',
   'scipy'
],
  version = '0.1.1',
  description = 'Reductions for Cost-Sensitive Multi-Class Classification',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/costsensitive',
  keywords = ['cost sensitive multi class', 'cost-sensitive multi-class classification', 'weighted all pairs', 'filter tree'],
  classifiers = [],
)