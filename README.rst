======================
 evoMPS \|. \|.:\|::>
======================
---------------------------------------------------------------
Quantum many-particle dynamics in 1D with matrix product states
---------------------------------------------------------------

Tutorial videos:

* Installation <http://vimeo.com/user19042101/evomps-tutorial-installation>
* Find ground state <http://vimeo.com/user19042101/evomps-tutorial-ground>
* Find excitations <http://vimeo.com/user19042101/evomps-tutorial-excitations>

Introduction
------------

evoMPS can find ground states and low-lying excited states of one-dimensional 
quantum systems (local Hamiltonians) directly in the thermodynamic limit of infinitely long chains.
States are represented as (infinite) matrix product states (MPS).

evoMPS finds ground states efficiently, even for critical systems, using the 
nonlinear conjugate gradient method. It computes dispersion relations using
the MPS tangent space as ansatz states for low-lying excitations with well-defined momenta.

evoMPS can also simulate nonuniform dynamics in a finite window of an infinite
system.

The above features set evoMPS apart from other MPS and DMRG software, which do
not usually work directly in the thermodynamic limit and do not exploit the MPS 
tangent space as an excitations ansatz.

evoMPS is based on algorithms published by Haegeman et al. [1] and Milsted et al. [2],
among others.

See the INSTALL file for installation instructions!

Key Features
------------

* Finds ground states and simulates dynamics of infinite systems
* Nonlinear conjugate gradient methods for finding ground states in infinite systems
* Calculates dispersion relations for infinite systems using tangent plane methods
* Handles locally nonuniform infinite systems (sandwich MPS aka infinite boundary conditions)
* Time-Dependent Variational Principle for simulating time evolution
* Runge-Kutta (RK4) and split-step (finite only) integrators for greater accuracy
* Supports local Hamiltonians: nearest-neighbour (or next-nearest neighbour)
* Limited support for long range Hamiltonians (in development, currently finite only using MPOs)

Usage
-----

The evoMPS algorithms are presented as python classes to be used in a script.
Some example scripts can be found in the "examples" directory.
To run an example script without installing the evoMPS modules, copy it to the base 
directory first e.g. under Windows::
    
    copy examples\transverse_ising_uniform.py .
    python transverse_ising_uniform.py

Essentially, the user defines a spin chain Hilbert space
and a nearest-neighbour Hamiltonian and then carries out a series of small 
time steps (numerically integrating the effective Schrödinger equation for the MPS parameters)::

    sim = EvoMPS_TDVP_Uniform(bond_dim, local_hilb_dim, my_hamiltonian)
    
    for i in range(max_steps):
        sim.update()
        
        my_exp_val = sim.expect_1s(my_op)
        
        sim.take_step_RK4(dtau)

Operators, including the Hamiltonian, are defined as arrays like this::

    pauli_z = numpy.array([[1, 0],
                           [0, -1]])

Calculating expectation values or other quantities can be done after each step 
as desired.

Switching between imaginary time evolution (for finding the ground state)
and real time evolution is as easy as multiplying the time step size by a factor of i!

Contact
-------

Please send comments to:

ashmilsted at <google's well-known email service>

To submit ideas or bug reports, please use the GitHub Issues system <http://github.com/amilsted/evoMPS/>.

References
----------

1. \J. Haegeman, J. I. Cirac, T. J. Osborne, I. Pizorn, H. Verschelde and F. Verstraete, arXiv:1103.0936. <http://arxiv.org/abs/1103.0936v2>.
2. \A. Milsted, T. J. Osborne, F. Verstraete, J. Haegeman, arXiv:1207.0691. <http://arxiv.org/abs/1207.0691>.

About
-----

evoMPS was originally developed as part of an MSc project by Ashley Milsted,
supervised by Tobias Osborne at the Institute for Theoretical Physics of
Leibniz Universität Hannover <http://www.itp.uni-hannover.de/>.

