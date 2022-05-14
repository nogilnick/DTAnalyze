import os
from   setuptools          import setup, Extension
from   subprocess          import check_output
from   distutils.sysconfig import get_python_inc
import numpy               as     np

incdir = os.path.join(get_python_inc(plat_specific=1))

# Check if gcc is installed
compileFlags = (['-O3', '-march=native'],
                ['/Ox'])

try:
   check_output(['gcc', '-v'])
   compileFlags = compileFlags[0]
except:  # Try cl flags
   compileFlags = compileFlags[1]
   
# Read in readme
with open('README.md') as F:
   desc = F.read()
   
module = Extension(
   'DTAnalyze.TreeAnalysis',
   include_dirs=[incdir, np.get_include()],
   libraries=[],
   library_dirs=[],
   sources=[os.path.join('DTAnalyze', 'TreeAnalysis.c')],
   extra_compile_args=compileFlags)

setup(
   name='TreeAnalysis',
   version='1.0',
   description='Python Decision Tree Analysis',
   author='Nicholas T. Smith',
   author_email='nicholastsmithblog@gmail.com',
   url="https://github.com/nicholastoddsmith/DTAnalyze",
   long_description_content_type="text/markdown",
   long_description=desc,
   packages=['TreeAnalysis'],
   ext_modules=[module],
   keywords=["Decision", "Tree", "RandomForest", "GBM", "Activation"],
   classifiers=[
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Natural Language :: English',
      'Operating System :: OS Independent',
      'Programming Language :: C',
      'Programming Language :: Cython',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3']
   )
