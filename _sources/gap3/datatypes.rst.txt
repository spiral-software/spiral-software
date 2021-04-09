	.. _gap_data_types:

Data Types
==========

.. only:: html

   .. contents:: :depth: 2

Scalars
+++++++

.. code-block:: none

	1; -1; 			# integer
	1.1; -1.543;		# IEEE floating point (GAP/Spiral specific)
	3/4;			# rational number
	Double(3/4);		# typecast to floating-point
	true; false;		# logical values
	Cplx(1, 1);		# complex number: 1 + i
	E(4);			# Cyclotomic numbers: complex root of unity

Basic Arithmetic/Logical Operations
+++++++++++++++++++++++++++++++++++

.. code-block:: none

	1 + 2; 1 + 2.0; 1 + 3/4;
	1.1 * Cplx(1, 1); 
	true and false; 
	Log2Int(8192);
	Re(E(4)); Im(E(4));	# Real and imaginary parts
	True; False;		# function returning true/false

Transcendental Functions
++++++++++++++++++++++++

.. code-block:: none

	SinPi(1.5);		# sin(1.5 * M_PI) -> floating-point number
	CosPi(3/4);		# cos(3/4*\pi) -> cyclotomic number	
	Sqrt(2);
	Sqrt(2.0);
	
Assignments
+++++++++++

.. code-block:: none

	x := 1;	 		# assign a <- 1
	x = 1;			# check equality
	x < 2; x > 0; x <> 7;	# inequalities
	
	Unbind(x);		# delete symbol
	let(a := 5, a+3);	# local symbol
	last;			# last result
	last2; last3;		# and even farther back

GAP Functions
+++++++++++++

.. code-block:: none

	a -> 1;	 		# maps a to 1
	a -> a + 1;		# maps a to a+1
	() -> 1;		# no argument, returns 1
	(a, b) -> a + b;	# adds the two arguments a and b
	f := (n) -> When(n = 1, 1, n * f(n - 1));	# recursive function
	ApplyFunc(f, [3]);	# apply function to parameter list


Strings
+++++++

Syntax
------

GAP strings are similar to C strings.

.. code-block:: none

	"";				# empty string
	'a';				# char
	"abc";				# string
	"a\tb\nc";			# control characters

Operations
----------

.. code-block:: none

	"abc"::"abc";			# concatenation
	Concat("abc", "abc");		# concatenation
	StringList("abc");		# string -> list of char
	StringToUpper("abc");		# -> upper case
	StringToLower("aBc");		# -> lower case
	StringInt(5);			# integer -> string
	

Lists
+++++

Syntax
------

.. code-block:: none

	[];				# empty list
	[1..4];				# dense list
	[1, 3, 4000];			# sparse list
	[[1, 2], [3, 4]];		# matrix = list of lists
	[[[1, 2], [3, 4]], [[5, 6], [7, 8]]];		# rank 3 tensor
	[1, rec(), Set([]), ()->3];	# mixed data type lists


Operations
----------

.. code-block:: none

	[1, 2]::[4..7];		# concatenation
	Concat([1, 2], [4..7]);	# concatenation
	List([1, 2, 4, 8], Log2Int);	# apply function to list elements 
	Map([1, 2, 4, 8], Log2Int);	# apply function to list elements 
	Length([1..5]);		# length of list
	1 + [2..5];			# pointwise arithmetic
	FoldL([1..3], (a,b)->a+b, 0);	# Fold left operation
	FoldR([1..3], (a,b)->a-b, 0);	# Fold right operation
	Cartesian([1..4], [3, 4]);	# Cartesian product
	ForAll([1..4], IsInt);		# Apply function, then reduce with AND
	ForAny([1..4], IsInt);		# Apply function, then reduce with OR
	Filtered([1..50], IsPrime);	# Apply filter function, drop if false
	1 in [1..5];			# membership test


Sets
++++

Syntax
------

.. code-block:: none

	Set([]);			# empty set
	Set([1, 2, 2, 3, 4]);		# Elements in sets are unique

Operations
----------

.. code-block:: none

	s := Set([]);	
	AddSet(s, 3);
	AddSet(s, 3);
	AddSet(s, 4);
	s;
	SubtractSet(s, [3]);
	s;
	

Vectors and Matrices
++++++++++++++++++++

Syntax
------

.. code-block:: none

	v := [1..3];				# list is a vector
	m := [[	1..3], [2..4], [3..5]];	# matrix

Operations
----------

.. code-block:: none

	m * v;					# matrix * vector
	v * m;					# vector * matrix
	TransposedMat(m);			# transpose matrix
	m * [[1],[2],[3]];			# column vector
	[[1..3]] * m;				# row vector


Records
+++++++

Syntax
------

.. code-block:: none

	rec();					# empty record
	rec(key1 := 2, key2 := 17);		# key/value pairs
	r := rec(a := 1, b := "2", 		# record of records
			 c := rec(a := 1), d:= []);	
	r.a;					# direct access
	i := "a";				# record field name
	r.(i);					# indirect access



Operations
----------

.. code-block:: none

	RecFields(r);				# List of record fields
	IsSystemRecField("__doc__");		# Some are system fields
	CopyFields(rec(a := 1, b := 2),	# merge records
	   rec(b := 3, c := 4));









