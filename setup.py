from distutils.core import setup, Extension
from evoMPS.version import __version__

from Cython.Distutils import build_ext

ext_modules = [Extension("evoMPS.matmul", ["evoMPS/matmul.py"]),
               Extension("evoMPS.tdvp_common", ["evoMPS/tdvp_common.pyx"]),
               Extension("evoMPS.tdvp_uniform", ["evoMPS/tdvp_uniform.py"])]

#for e in ext_modules:
#    e.pyrex_directives = {"profile": True}

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
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules
      )
