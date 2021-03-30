.. _introduction:

Introduction
============

What is SPIRAL?
---------------

SPIRAL is a program generation system for linear transforms and other mathematical functions that produces very high performance code for a 
wide spectrum of hardware platforms.  The input to SPIRAL is high level mathematical algorithm specifications along with selected architectural 
and micro-architectural parameters. The output is performance-optimized code in a language such as C, possibly augmented with vector 
intrinsics and threading instructions.  It is a mature system embodying over two decades of research and development at Carnegie Mellon University,   
with a long pedigree of publications, large successful projects and commercial use for vendor math libraries, and it continues to be an active 
focal point for research at CMU.

SPIRAL focuses on the components of a software system that generally do not port well between different platforms -- computationally intensive 
algorithms that traditionally require expert platform-specific tuning to achieve good performance.  SPIRAL lets a developer express algorithms 
at a mathematical abstraction above the code level, and it generates the appropriate code for a given target.  Similar to how high-level programming 
languages replaced assembly language for most coding tasks, the higher level mathematical abstractions can replace standard source code for 
performance-sensitive kernels.


Why SPIRAL?
-----------

The central motivator behind SPIRAL is that modern computational platforms are so complex, both in depth and breadth, that 
manually writing critical high-performance software for those systems has become an increasingly impractical, if not intractable,
undertaking.  And even if successfully written and tuned for one platform, the performance probably doesn't carry over as machines
are replaced every few years.  Every new combination of nodes, cores, threads, memory speeds, cache hierarchies, instruction sets,
registers, and so on has its own idiosyncrasies.  A popular and dependable numerically intensive package may move several times
in its lifetime, often having to perform well below its potential because of prohibitive costs to rework it.

For someone who hasn't had to fine tune production high-performance code for a modern hardware platform there are a lot
of non-obvious gotchas that make the problem much messier than it looks from the outside.  A well written overview of the issues, 
:ref:`How To Write Fast Numerical Code: A Small Introduction <fastnumericalcode>`, is a valuable introduction to the many-faceted problem.



License
-------

SPIRAL is a fully open source project publicly available under a BSD-style license.  This means you are free to use SPIRAL for whatever you like, be it academic, commercial, 
creating forks or derivatives, as long as you copy the license statement if you redistribute it (see the `LICENSE <https://github.com/spiral-software/spiral-software/blob/master/LICENSE>`__ file for details).

Though not required by the SPIRAL license, if it is convenient for you, please cite SPIRAL when using it in your work and also consider contributing 
your changes back to the main SPIRAL project.


Citation
--------

To cite SPIRAL in publications use

    | Franz Franchetti, Tze Meng Low, Doru Thom Popovici, Richard M. Veras, Member, 
    | Daniele G. Spampinato, Jeremy R. Johnson, Markus Püschel, James C. Hoe, José M. F. Moura:
    | **SPIRAL: Extreme Performance Portability**.
    | Proceedings of the IEEE, Special Issue on *From High Level Specifications to High Performance Code*, 2018

A BibTeX entry for LaTeX users is

.. code-block:: none

	@ARTICLE{Franchetti:18,
	 AUTHOR = {Franz Franchetti and Tze-Meng Low and Thom Popovici and Richard Veras and Daniele G. Spampinato and Jeremy Johnson and Markus P{\"u}schel and James C. Hoe and Jos{\'e} M. F. Moura},
	 TITLE = {{SPIRAL}: Extreme Performance Portability},
	 JOURNAL = {Proceedings of the IEEE, special issue on ``From High Level Specification to High Performance Code''},
	 VOLUME = {106},
	 NUMBER = {11},
	 YEAR = {2018}
	}





