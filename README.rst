========
 evoMPS
========
---------------------------------------------------------------
Quantum many-particle dynamics in 1D with matrix product states
---------------------------------------------------------------

Introduction
------------

evoMPS simulates time-evolution (real or imaginary time) of one-dimensional 
many-particle quantum systems using matrix product states
(MPS) and the time dependent variational principle (TDVP).

It can be used to efficiently find ground states and simulate dynamics.

The evoMPS implementation assumes a nearest-neighbour Hamiltonian one of the 
following situations:

* states on a finite chain with open boundary conditions
* spatially uniform states on an infinite chain
* otherwise uniform states with a localized nonuniformity on an infinite chain

It is based on algorithms published by: 

* Jutho Haegeman
* \J. Ignacio Cirac
* Tobias J. Osborne
* Iztok Pizorn
* Henri Verschelde
* Frank Verstraete

and available on arxiv.org under arXiv:1103.0936v2 [cond-mat.str-el]
<http://arxiv.org/abs/1103.0936v2>. The algorithm for handling localized
nonuniformities on infinite chains was developed by:

* Ashley Milsted
* Tobias J. Osborne
* Frank Verstraete
* Jutho Haegeman

For details, see doc/implementation_details.pdf and the source code itself,
which I endeavour to annotate thoroughly.

evoMPS is implemented in Python using Scipy <http://www.scipy.org> and
benefits from optimized linear algebra libraries being installed (BLAS and LAPACK).
For more details, see INSTALL.

evoMPS is being developed as part of an MSc project by Ashley Milsted,
supervised by Tobias Osborne at the Institute for Theoretical Physics of
Leibniz Universität Hannover <http://www.itp.uni-hannover.de/>.

Usage
-----

The evoMPS algorithms are presented as python classes to be used in a script.
Some example scripts can be found in the "examples" directory.

Essentially, the user defines a site Hilbert space dimension
and a nearest-neighbour Hamiltonian (as a python function
or "callable") in terms of local Hilbert space matrix elements,
then carries out a series of small time steps (numerically
integrating the "Schrödinger equation" for the MPS parameters), calculating
expectation values or other quantities after each step as desired.

Switching between imaginary time evolution (for finding the ground state)
and real time evolution is as easy as multiplying the time step size by a factor of i!


Contact
-------

Please send bug reports or comments to:

ashmilsted at <google's well-known email service>
