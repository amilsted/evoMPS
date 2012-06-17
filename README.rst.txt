========
 evoMPS
========

Introduction
------------

evoMPS simulates time-evolution (real or imaginary time) of one-dimensional 
many-particle quantum systems using matrix product states
(MPS) and the time dependent variational principle (TDVP).

It can be used to efficiently find ground states and simulate dynamics.

The evoMPS implementation assumes a nearest-neighbour Hamiltonian and either

* a finite lattice with open boundary conditions or
* spatially uniform states on an infinite lattice.

It is based on algorithms published by: 

* Jutho Haegeman
* J. Ignacio Cirac
* Tobias J. Osborne
* Iztok Pizorn
* Henri Verschelde
* Frank Verstraete

and available on arxiv.org under arXiv:1103.0936v2 [cond-mat.str-el]
<http://arxiv.org/abs/1103.0936v2>.

For details, see doc/implementation_details.pdf and the source code itself,
which I endeavour to annotate thoroughly.

evoMPS is being developed as part of an MSc project by Ashley Milsted,
supervised by Tobias Osborne at the Institute for Theoretical Physics of
Leibniz Universität Hannover <http://www.itp.uni-hannover.de/>.


Contact
-------

Please send bug reports or comments to:

ashmilsted at <google's well-known email service>
