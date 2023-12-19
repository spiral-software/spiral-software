.. _gettingstarted:

Getting Started
===============

.. contents::


.. _installing:

Installing SPIRAL
+++++++++++++++++

SPIRAL source is `available on GitHub <https://github.com/spiral-software/spiral-software>`__ 
under a non-viral `BSD-style license <https://github.com/spiral-software/spiral-software/blob/master/LICENSE>`__.  It builds and runs on several platforms, 
including Windows, Linux, macOS, and Raspberry Pi.

https://github.com/spiral-software/spiral-software
  
See the `README on GitHub <https://github.com/spiral-software/spiral-software/blob/master/README.md>`__ for more information.


	
GAP and the Command Line
++++++++++++++++++++++++

The input language to SPIRAL is an extended version of `GAP 3 <http://www.gap-system.org/Gap3/gap3.html>`__.  

Chapters 1 and 2 of the `GAP Manual <https://www.spiral.net/doc/pdf/GAP_Manual.pdf>`__ cover most of what you need for SPIRAL.


Basic Syntax
------------

In a nutshell:

* Whitespace is ignored
* Statements can span multiple lines and end with a semicolon (**;**)
* Blocks of statements are delineated with keywords, like do/od, if/then/else/fi
* Variable names are case sensitive and are declared implicitly by use
* **:=** is the assignment operator and **=** is the boolean equals operator


Command Line
------------

When you start SPIRAL in a terminal window (``> spiral``), SPIRAL displays the **spiral>** command
prompt that lets you enter individual statements interactively.  To exit, enter **quit;**.

The SPIRAL command line allows edits similar to the Linux terminal window and the Windows command window.

* **CTRL+E** moves the cursor to the end of the line.
* **CTRL+A** moves the cursor to the beginning of the line.
* **Left** and **Right** arrows move the cursor one character on the current line.
* **Up** and **Down** arrows scroll back and forth through previous entries.

These have special meaning to SPIRAL:

* **CTRL+C** interrupts a running statement
* **CTRL+D** is equivalent to **quit;**
* **TAB** completes a command up to a point of ambiguity.  Pressing **TAB** again shows all choices.
* **CTRL+W** shows all the field names for a record.

SPIRAL stores entries from previous sessions, so work from previous sessions is available when you
start SPIRAL.  Note that the history file is only saved if you exit SPIRAL with the **quit;** command.

Batch Mode
----------

You can start SPIRAL in batch mode by specifying an input file at startup using the standard I/O redirection syntax,
for example::

	spiral < myscript.txt
	
SPIRAL will run the script and exit.  Direct the output to a file to save it::

	spiral < myscript.txt > myresults.txt


Configuration
+++++++++++++

SPIRAL Options Record
---------------------

.. code-block:: none

	opts := SpiralDefaults;
	opts.globalUnrolling;
	opts.arrayBufModifier;
	opts.includes;
	opts.unparser;
	opts.codegen;
	opts.useDeref; 
	opts.Xtype;
	opts.compileStrategy;
	opts.breakdownRules;
	...


Local Configuration
-------------------

Set via _spiral.g

.. code-block:: none

	LocalConfig;
	LocalConfig.getOpts(SSE_4x32f);
	LocalConfig.cpuinfo;
	LocalConfig.gapinfo;
	LocalConfig.osinfo;


	