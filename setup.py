from numpy.distutils.core import setup, Extension
from evoMPS.version import __version__
import numpy as np

from distutils.command.build import build as _build

ext_modules = [Extension("evoMPS.matmul", ["evoMPS/matmul.c"]),
               Extension("evoMPS.core_common", ["evoMPS/core_common.c"]),
               Extension("evoMPS.allclose", ["evoMPS/allclose.c"]),
               Extension("evoMPS.tdvp_calc_C", ["evoMPS/tdvp_calc_C.c"]),
               Extension("evoMPS.expokit", ["evoMPS/expokit/expokit.f", 
                                            "evoMPS/expokit/expokit.pyf"],
                         libraries = ['blas', 'lapack'])
              ]

class build(_build):         
    def initialize_options(self):
        _build.initialize_options(self)
        self.disable_build_ext = 0
        
    def finalize_options(self):
        _build.finalize_options(self)
        
        if self.disable_build_ext != 0:
            self.sub_commands.remove(self.sub_commands[2])

build.user_options += [
            ('disable-build-ext', None,
             "disable build-ext and use pure python mode")]

setup(cmdclass={'build': build},
      name='evoMPS',
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
