from distutils.core import setup, Extension
from evoMPS.version import __version__

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
      )
