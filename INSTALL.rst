===================
 Installing evoMPS
===================

Obtaining the Latest Software
-----------------------------

The latest version of evoMPS can be downloaded from 
GitHub <http://github.com/amilsted/evoMPS>.


Prerequisites
-------------

On Windows, an easy way to obtain everything required is to download and
install a numerics-oriented Python distribution such as

* Anaconda <https://www.continuum.io/why-anaconda> 
* enthought python distribution <https://www.enthought.com/products/canopy/>
* pythonxy <https://python-xy.github.io/>

The full installation of either of these includes everything you need.
Otherwise, the following are required:

* Python 3 <http://www.python.org> (tested on Python 3.6)
* Numpy <http://numpy.scipy.org> (tested on 1.14.0)
* Scipy <http://www.scipy.org> (version 0.7.0 or newer - tested on 1.0.0)

Numpy/Scipy should be compiled with a LAPACK library, preferably
an optimized one such as the MKL from Intel <https://software.intel.com/en-us/intel-mkl>
(now included for free with many python distributions due to licensing changes by Intel)
or OpenBLAS <http://www.openblas.net/>.

If present, a c compiler may be used to compile some parts of evoMPS,
resulting in performance gains, especially at low bond dimensions.

To setup a c compiler on Windows, see <https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>.

To run the included examples, the following is also required:

* matplotlib <http://matplotlib.sourceforge.net/> (tested on 1.1.0)


Building and Installation
-------------------------

To install the evoMPS package, go to the source directory and run::

    python setup.py install

Alternatively, to install for the current user only, run::

    python setup.py install --user 

Installation is not strictly necessary, as scripts using evoMPS can
also be run from the base source directory.

As of version 1.9, setup.py does *not* perform any compilation of extensions.
This makes it easier to install the pure python version in case c or FORTRAN
compilers are not present. To compile extensions, also run::

    python setup_ext.py install
    
Again, add the --user option to install for the current user only.

To also compile pure c versions of the transfer-operator maps (with additional
support for OpenMP threading), run::

    python setup_ext_pure_c.py install


Getting Started
---------------

Examples have been provided in the examples/ subdirectory. After installing
evoMPS as described above, they can be run using e.g.::

    python transverse_ising.py

To run an example without installing, copy it to the base source directory first.
