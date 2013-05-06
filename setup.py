from distutils.core import setup, Extension
from evoMPS.version import __version__
import numpy as np

ext_modules = [Extension("evoMPS.matmul", ["evoMPS/matmul.c"]),
               Extension("evoMPS.tdvp_common", ["evoMPS/tdvp_common.c"]),
               Extension("evoMPS.allclose", ["evoMPS/allclose.c"]),
               Extension("evoMPS.tdvp_calc_C", ["evoMPS/tdvp_calc_C.c"])]

setup(name='evoMPS',
      version=__version__,
      description='Python scripts for simulating time evolution of matrix product states',
      author='Ashley Milsted',
      url='https://github.com/amilsted/evoMPS',
      license="BSD",
      classifiers=[
        'Environment :: Console',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
      ],      
      packages = ['evoMPS'],
      requires = ["scipy (>=0.7)"],
      include_dirs = [np.get_include()],
      ext_modules = ext_modules
      )
