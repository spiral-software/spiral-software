
.. _bibliography:

Further Reading
===============

This is a short annotated list of papers and presentations about SPIRAL.  For a comprehensive list
see Carnegie Mellon's `SPIRAL Publications <http://spiral.ece.cmu.edu:8080/pub-spiral>`__ page.



About SPIRAL in General
-----------------------


.. _spiraliee18:

SPIRAL: Extreme Performance Portability (2018)
++++++++++++++++++++++++++++++++++++++++++++++

This is the most recent major paper about SPIRAL.

    | Franz Franchetti, Tze Meng Low, Doru Thom Popovici, Richard M. Veras, 
    | Daniele G. Spampinato, Jeremy R. Johnson, Markus Püschel, James C. Hoe, José M. F. Moura:
    | **SPIRAL: Extreme Performance Portability**.
    | Proceedings of the IEEE, Special Issue on *From High Level Specifications to High Performance Code*, 2018

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/SPIRAL_IEEE_2018.pdf>`__  `[ BibTeX ] <http://spiral.ece.cmu.edu:8080/pub-spiral/bibtextFile/bibtext_299.html>`__


.. _spiralenc11:

SPIRAL (2011)
+++++++++++++

If you are only going to read one article about SPIRAL, this is the one to read.  It walks
through the major concepts and components of SPIRAL with lots of illustrations and examples.

    | Markus Püschel, Franz Franchetti, Yevgen Voronenko:
    | **Spiral**.
    | Encyclopedia of Parallel Computing 2011: 1920-1933

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/spiral-enc11.pdf>`__ `[ BibTeX ] <http://spiral.ece.cmu.edu:8080/pub-spiral/bibtextFile/bibtext_146.html>`__
	

.. _spiral10slides:

Spiral: Computer Generation of Performance Libraries (2011)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This short slide presentation illustrates the key points about SPIRAL.

    | José M. F. Moura, Markus Püschel, Franz Franchetti:
    | **Spiral: Computer Generation of Performance Libraries**
    | Carnegie Mellon University 2011

.. only:: html

    | `[ Slides ] <https://www.spiral.net/doc/slides/spiral-10slides.pdf>`__


.. _spiraliee05:

SPIRAL: Code Generation for DSP Transforms (2005)
+++++++++++++++++++++++++++++++++++++++++++++++++	
	
Though several papers about SPIRAL predate it, this 2005 deep dive is SPIRAL's historical debut on the main stage.  SPIRAL has grown a lot since then,
but this article covers most of major components, particularly *Signal Processing Language (SPL)* and *Breakdown Rules*.  As a bonus, it has photos of 
the (young) authors/team.

    | Markus Püschel, José M. F. Moura, Jeremy R. Johnson, David A. Padua, Manuela M. Veloso, Bryan Singer, 
    | Jianxin Xiong, Franz Franchetti, Aca Gacic, Yevgen Voronenko, Kang Chen, Robert W. Johnson, Nicholas Rizzolo:
    | **SPIRAL: Code Generation for DSP Transforms**.
    | Proceedings of the IEEE 93(2): 232-275 (2005)

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/IEEE_2005.pdf>`__ `[ BibTeX ] <http://spiral.ece.cmu.edu:8080/pub-spiral/bibtextFile/bibtext_1.html>`__





FFTs and Related Transforms
---------------------------


.. _fftenc11:

Fast Fourier Transform (2011)
+++++++++++++++++++++++++++++

This well-written and illustrated article is a must-read for someone wanting to learn about SPIRAL.  It explains the linear algebra
representation of FFTs foundational to SPIRAL in the context of several FFT algorithms.

    | Franz Franchetti, Markus Püschel:
    | **FFT (Fast Fourier Transform)**.
    | Encyclopedia of Parallel Computing 2011: 658-671

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/fft-enc11.pdf>`__
	


.. _prunedfft09:

Generating High Performance Pruned FFT Implementations (2009)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This short paper is an interesting example of a specific adaption of SPIRAL.

    | Franz Franchetti, Markus Püschel:
    | **Generating High Performance Pruned FFT Implementations**.
    | ICASSP 2009: 549-552

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/icassp09_PrunedDFT.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/icassp09.pdf>`__



.. _fastnumericalcode:

How To Write Fast Numerical Code: A Small Introduction (2007)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This practical tutorial introduces a set of techniques to improve numerical code performance, focusing 
on optimizations for memory hierarchy.

    | S. Chellappa, F. Franchetti, and M. Püschel
    | **How To Write Fast Numerical Code: A Small Introduction**
    | Proceedings of the Generative and Transformational Techniques in Software Engineering (GTTSE) 2007	

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/FastNumericalCode.pdf>`__
	

Key Internal Functionality
--------------------------



.. _rewriting06:

A Rewriting System for the Vectorization of Signal Transforms (2006)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Understanding SPIRAL requires understanding its layers of rewriting.  This paper shows how SPIRAL
uses rewrite rules to map signal transforms to vector instructions.

    | F. Franchetti, Y. Voronenko, M. Püschel
    | **A Rewriting System for the Vectorization of Signal Transforms**
    | Proceedings High Performance Computing for Computational Science (VECPAR) 2006, LNCS 4395, pages 363-377

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/vecpar06.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/vecpar06-franchetti.pdf>`__

	

.. _pldi05:
	
Formal Loop Merging for Signal Transforms (2005)
++++++++++++++++++++++++++++++++++++++++++++++++

This key paper introduces **Sigma SPL**, which was a new level of abstraction added to SPIRAL in 2005.  Another
must-read, this paper and its companion slide deck are well worth careful study.

    | Franz Franchetti, Yevgen Voronenko, Markus Püschel:
    | **Formal Loop Merging for Signal Transforms**. 
    | PLDI 2005: 315-326

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/Formal_Loop_Merging_pldi05.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/Formal_Loop_Merging_Slides_pldi05.pdf>`__



Vector Instructions and Other Parallelism
-----------------------------------------

.. _ffte-spiral:

FFTE on SVE: SPIRAL-Generated Kernels (2020)
++++++++++++++++++++++++++++++++++++++++++++

Using SPIRAL to generate vectorized FFT kernels for 
ARM Scalable Vector Extension (SVE) produced significant
speedups over automatic vectorization at compile time.

    | Daisuke Takahashi, Franz Franchetti
    | **FFTE on SVE: SPIRAL-Generated Kernels**
    | International Conference on High Performance Computing in Asia-Pacific Region (HPCAsia), 2020

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/ffte-spiral.pdf>`__


.. _ipdps18:

Large Bandwidth-Efficient FFTs on Multicore and Multi-Socket Systems (2018)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This paper illustrates SPIRAL's state of the art for exploiting multiple levels of 
parallelism in new generation processors, including splitting compute and memory
access functions to different hardware threads.

    | D. T. Popovici, T. M. Low, F. Franchetti
    | **Large Bandwidth-Efficient FFTs on Multicore and Multi-Socket Systems**
    | IEEE International Parallel & Distributed Processing Symposium (IPDPS), 2018

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/ipdps2018_dtp.pdf>`__
	



.. _ipdps15:

Generating Optimized Fourier Interpolation Routines for Density Functional Theory Using SPIRAL (2015)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is a detailed example of using a collection of techniques in SPIRAL for big performance gains in
upsampling large multi-dimensional data sets.
	
    | D. A. Popovici, F. Russell, K. Wilkinson, C-K. Skylaris, P. H. J. Kelly, F. Franchetti
    | **Generating Optimized Fourier Interpolation Routines for Density Functional Theory Using SPIRAL**
    | 29th International Parallel & Distributed Processing Symposium (IPDPS), 2015	

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/ipdps15.pdf>`__	



.. _pldi13:

When Polyhedral Transformations Meet SIMD Code Generation (2013)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Looking at the general problem of loop optimization, this paper shows how SPIRAL's SIMD code generation
can work together with a polyhedral framework.

    | M. Kong, R. Veras, K. Stock, F. Franchetti, L.-N. Pouchet, and P. Sadayappan
    | **When Polyhedral Transformations Meet SIMD Code Generation**
    | ACM SIGPLAN PLDI, 2013

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/pldi13.pdf>`__




.. _hpcgc12:

Automatic Generation of the HPC Challenges Global FFT Benchmark for BlueGene/P (2012)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In 2010 code generated by SPIRAL was part of IBM's winning submission for the
2010 HPC Challenge Class II.  SPIRAL generated a single giant 1D FFT that 
spanned up to 128K cores.
	
    | F. Franchetti, Y. Voronenko, and G. Almasi
    | **Automatic Generation of the HPC Challenges Global FFT Benchmark for BlueGene/P**
    | In Proceedings of High Performance Computing for Computational Science (VECPAR) 2012

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/AGofHPC.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/vecpar2012.pdf>`__

	


.. _ics11:

Automatic SIMD Vectorization of Fast Fourier Transforms for the Larrabee and AVX Instruction Sets (2011)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This paper on the successful inclusion of AVX support in SPIRAL was written the same year that the AVX 
instruction set became available.

    | Daniel McFarlin, Volodymyr Arbatov, Franz Franchetti and Markus Püschel
    | **Automatic SIMD Vectorization of Fast Fourier Transforms for the Larrabee and AVX Instruction Sets**
    | Proc. International Conference on Supercomputing (ICS), 2011

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/ics2011.pdf>`__




.. _spmag09:

Discrete Fourier Transform on Multicores: Algorithms and Automatic Implementation (2009)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This paper discusses problems and solutions to splitting an FFT implementation across
multiple cores.

    | F. Franchetti, Y. Voronenko, S. Chellappa, J. M. F. Moura, and M. Püschel
    | **Discrete Fourier Transform on Multicores: Algorithms and Automatic Implementation**
    | IEEE Signal Processing Magazine, special issue on “Signal Processing on Platforms with Multiple Cores”, 2009.

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/spmag09.pdf>`__




Generating Non-Transform Kernels
--------------------------------



.. _has17:

High-Assurance SPIRAL:  End-to-End Guarantees for Robot and Car Control (2017)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This feature article showcases a novel application of SPIRAL in the DARPA
High Assurance Cyber Military Systems (HACMS) program.

    | F. Franchetti, T. M. Low, S. Mitsch, J. P. Mendoza, L. Gui, A. Phaosawasdi, D. Padua, S. Kar, J. M. F. Moura, M. Franusich, J. Johnson, A. Platzer, and M. Veloso
    | **High-Assurance SPIRAL:  End-to-End Guarantees for Robot and Car Control**
    | IEEE Control Systems Magazine, 2017, pages 82-103.

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/has2017.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/hacms-overview.pdf>`__



.. _viterbi10:

Computer Generation of Efficient Software Viterbi Decoders
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    | F. de Mesmay, S. Chellappa, F. Franchetti and M. Püschel
    | **Computer Generation of Efficient Software Viterbi Decoders**
    | Proceedings of International Conference on High-Performance Embedded Architectures and Compilers (HIPEAC), 2010

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/hipeac2010.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/hipeac2010.pdf>`__


.. _ol09:

Operator Language: A Program Generation Framework for Fast Kernels (2009)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Winning a *Best Paper Award*, this describes **Operator Language**, which empowers
SPIRAL to generate non-transform code.

    | Franz Franchetti, Frédéric de Mesmay, Daniel S. McFarlin, Markus Püschel:
    | **Operator Language: A Program Generation Framework for Fast Kernels**. 
    | DSL 2009: 385-409

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/OperatorLanguage.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/dslwc09.pdf>`__




.. _sar09:

High Performance Synthetic Aperture Radar Image Formation On Commodity Architectures (2009)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In 2009 the SPIRAL folks at CMU figured out how to generate really good Synthetic Aperture Radar (SAR)
code.

    | D. McFarlin, F. Franchetti, M. Püschel and J.M.F. Moura
    | **High Performance Synthetic Aperture Radar Image Formation On Commodity Architectures**.
    | Proceedings of SPIE Conference on Defense, Security, and Sensing, 2009

.. only:: html

    | `[ PDF ] <https://www.spiral.net/doc/papers/SAR-spie09.pdf>`__ `[ Slides ] <https://www.spiral.net/doc/slides/spiesar09.pdf>`__

|

****
	
*Copyrights to many of the above papers are held by the publishers. 
The attached PDF files are preprints. It is understood that all persons 
copying this information will adhere to the terms and constraints invoked 
by each author's copyright. These works may not be reposted without the explicit 
permission of the copyright holder.*


