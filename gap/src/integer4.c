/****************************************************************************
**
*W  integer4.c                  GAP source                   Martin Schoenert
**                                                           & Alice Niemeyer
**                                                           & Werner  Nickel
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file implements the  functions  handling  arbitrary  size  integers.
**
**  There are three integer types in GAP: 'T_INT', 'T_INTPOS' and 'T_INTNEG'.
**  Each integer has a unique representation, e.g., an integer  that  can  be
**  represented as 'T_INT' is never  represented as 'T_INTPOS' or 'T_INTNEG'.
**
**  'T_INT' is the type of those integers small enough to fit into  29  bits.
**  Therefor the value range of this small integers is: $-2^{28}...2^{28}-1$.
**  This range contains about 99\% of all integers that usually occur in GAP.
**  (I just made up this number, obviously it depends on the application  :-)
**  Only these small integers can be used as index expression into sequences.
**
**  Small integers are represented by an immediate integer handle, containing
**  the value instead of pointing  to  it,  which  has  the  following  form:
**
**      +-------+-------+-------+-------+- - - -+-------+-------+-------+
**      | guard | sign  | bit   | bit   |       | bit   | tag   | tag   |
**      | bit   | bit   | 27    | 26    |       | 0     | 0     | 1     |
**      +-------+-------+-------+-------+- - - -+-------+-------+-------+
**
**  Immediate integers handles carry the tag 'T_INT', i.e. the last bit is 1.
**  This distuingishes immediate integers from other handles which  point  to
**  structures aligned on 4 byte boundaries and therefor have last bit  zero.
**  (The second bit is reserved as tag to allow extensions of  this  scheme.)
**  Using immediates as pointers and dereferencing them gives address errors.
**
**  To aid overflow check the most significant two bits must always be equal,
**  that is to say that the sign bit of immediate integers has a  guard  bit.
**
**  The macros 'INTOBJ_INT' and 'INT_INTOBJ' should be used to convert  between
**  a small integer value and its representation as immediate integer handle.
**
**  'T_INTPOS' and 'T_INTNEG' are the types of positive  respective  negative
**  integer values  that  can  not  be  represented  by  immediate  integers.
**
**  This large integers values are represented in signed base 65536 notation.
**  That means that the bag of  a  large  integer  has  the  following  form:
**
**      +-------+-------+-------+-------+- - - -+-------+-------+-------+
**      | digit | digit | digit | digit |       | digit | digit | digit |
**      | 0     | 1     | 2     | 3     |       | <n>-2 | <n>-1 | <n>   |
**      +-------+-------+-------+-------+- - - -+-------+-------+-------+
**
**  The value of this  is:  $d0 + d1 65536 + d2 65536^2 + ... + d_n 65536^n$,
**  respectivly the negative of this if the type of this object is T_INTNEG'.
**
**  Each digit is  of  course  stored  as  a  16  bit  wide  unsigned  short.
**  Note that base 65536 allows us to multiply 2 digits and add a carry digit
**  without overflow in 32 bit long arithmetic, available on most processors.
**
**  The number of digits in every  large  integer  is  a  multiple  of  four.
**  Therefor the leading three digits of some values will actually  be  zero.
**  Note that the uniqueness of representation implies that not four or  more
**  leading digits may be zero, since |d0|d1|d2|d3| and |d0|d1|d2|d3|0|0|0|0|
**  have the same value only one, the first, can be a  legal  representation.
**
**  Because of this it is possible to do a  little  bit  of  loop  unrolling.
**  Thus instead of looping <n> times, handling one digit in each  iteration,
**  we can loop <n>/4 times, handling  four  digits  during  each  iteration.
**  This reduces the overhead of the loop by a factor of  approximatly  four.
**
**  Using base 65536 representation has advantages over  using  other  bases.
**  Integers in base 65536 representation can be packed  dense  and  therefor
**  use roughly 20\% less space than integers in base  10000  representation.
**  'SumInt' is 20\% and 'ProdInt' is 40\% faster for 65536 than  for  10000,
**  as their runtime is linear respectivly quadratic in the number of digits.
**  Dividing by 65536 and computing the remainder mod 65536 can be done  fast
**  by shifting 16 bit to  the  right  and  by  taking  the  lower  16  bits.
**  Larger bases are difficult because the product of two digits will not fit
**  into 32 bit, which is the word size  of  most  modern  micro  processors.
**  Base 10000 would have the advantage that printing is  very  much  easier,
**  but 'PrInt' keeps a terminal at 9600 baud busy for almost  all  integers.
*/

#include        "system.h"              /* GAP3 system package             */
#include        "objects.h"
#include		"string4.h"
#include        "memmgr.h"              /* garbage collector               */
#include        "eval.h"                /* generic operations              */
#include        "gstring.h"              /* strings                         */
#include        "scanner.h"             /* Pr()                            */
#include        "integer.h"
#include        <stdio.h>
#include        <stdlib.h>
#include        "integer4.h"            /* integers                        */

/* GAP3 only has Error() */
#define ErrorReturnObj(err_msg, arg1, arg2, msg) \
     Error(err_msg " " msg, arg1, arg2);


/****************************************************************************
**
*T  TypDigit  . . . . . . . . . . . . . . . . . . . .  type of a single digit
**
**  'TypDigit' is the type of a single digit of an  arbitrary  size  integer.
**  This is of course unsigned short int, which gives us the 16 bits we want.
**
**  'TypDigit' is defined in the declaration file of the package as follows:
**
#ifdef SYS_IS_64_BIT
typedef UInt4           TypDigit;
#else
typedef UInt2           TypDigit;
#endif
#define NR_DIGIT_BITS      (8 * sizeof(TypDigit))
#define INTBASE            (1L << NR_DIGIT_BITS)
#define NR_SMALL_INT_BITS  (2*NR_DIGIT_BITS - 4)
*/

#define SIZE_INT(op)    (SIZE_OBJ(op) / sizeof(TypDigit))
#define ADDR_INT(op)    ((TypDigit*)PTR_BAG(op))


/****************************************************************************
**
*F  EvInt( <hdInt> )  . . . . . . . . . . . . .  evaluate an integer constant
**
**  'EvInt' returns  the value  of the  integer  <hdInt>.  Since integers are
**  constants and thus  selfevaluating this simply  returns <hdInt>.  This is
**  the evaluation function for the types 'T_INT', 'T_INTPOS', 'T_INTNEG'.
*/
Obj		EvInt(Obj integer)
{
	return integer;
}


/****************************************************************************
**
*F  PrInteger( <int> ) . . . . . . . . . . . . . . print an integer constant
**
**  'PrInteger'  prints  the integer  <int>   in the  usual  decimal notation.
**  'PrInteger' handles objects of type 'T_INT', 'T_INTPOS' and 'T_INTNEG'.
**
**  Large integers are first converted into  base  10000  and  then  printed.
**  The time for a conversion depends quadratically on the number of  digits.
**  For 2000 decimal digit integers, a screenfull,  it  is  reasonable  fast.
**
**  The number  of digits  needed in PrIntD[] is the ceiling of the logarithm
**  with respect to base PRINT_BASE of
**
**           ( (1<<NR_DIGIT_BITS) )^1000 - 1.
**
**  The latter is the largest number that can be represented with 1000 digits
**  of type TypDigit.
**
**  If NR_DIGIT_BITS is 16, we get 1205.
**  If NR_DIGIT_BITS is 32, we get 1071.
**
**  The subsidiary function IntToPrintBase converts an integer into base
**  PRINT_BASE, leaving the result in base PrIntD. It returns the index of the
**  most significant digits. It assumes that the argument is a large
**  integer small enough to fit.
*/

TypDigit        PrIntC[1000];          /* copy of integer to be printed   */


#ifdef SYS_IS_64_BIT
#ifndef WIN64
#define PRINT_BASE 1000000000L          /* 10^9                            */
#else
#define PRINT_BASE 1000000000LL			/* 10^9                            */
#endif
#define PRINT_FORMAT "%09d"             /* print 9 decimals at a time      */
#define CHARS_PER_PRINT_BASE 9
TypDigit        PrIntD[1071];          /* integer converted to base 10^9  */
#define NR_HEX_DIGITS 8

#else

#define PRINT_BASE 10000
#define PRINT_FORMAT "%04d"             /* print 4 decimals at a time      */
#define CHARS_PER_PRINT_BASE 4
TypDigit        PrIntD[1205];          /* integer converted to base 10000 */
#define NR_HEX_DIGITS 4

#endif


Int		IntToPrintBase(Obj op)
{
	UInt		 i = 0;				 /* loop counter                    */
	UInt		 k;		         /* loop counter                    */
	UInt         c;              /* carry in division step          */
	TypDigit	*p;              /* loop pointer                    */


	for (k = 0; k < SIZE_INT(op); k++)
	{
		PrIntC[k] = ADDR_INT(op)[k];
	}

	while (k > 0 && PrIntC[k - 1] == 0)
	{
		k--;
	}

	while (k > 0) 
	{
		for (c = 0, p = PrIntC + k - 1; p >= PrIntC; p--) 
		{
			c = (c << NR_DIGIT_BITS) + *p;
			*p = (TypDigit)(c / PRINT_BASE);
			c = c - PRINT_BASE * *p;
		}

		PrIntD[i++] = (TypDigit)c;

		while (k > 0 && PrIntC[k - 1] == 0)
		{
			k--;
		}
	}

	return i - 1;
}

void	PrInteger(STREAM stream, Obj op, int indent)
{
	Int		i;           /* loop counter                    */

	/* print a small integer                                               */
	if (IS_INTOBJ(op)) 
	{
		//**INDENT** Pr("%>%d%<", INT_INTOBJ(op), 0);
		SyFmtPrint(stream, "%d", INT_INTOBJ(op));
	} 
	/* print a large integer                                               */
	else if (SIZE_INT(op) < 1000) 
	{

		/* start printing, %> means insert '\' before a linebreak          */
		//**INDENT** Pr("%>", 0, 0);


		if (TNUM_OBJ(op) == T_INTNEG)
		{
			//Pr("-", 0, 0);
			SyFmtPrint(stream, "-");
		}

		/* convert the integer into base PRINT_BASE                        */
		i = IntToPrintBase(op);

		/* print the base PRINT_BASE digits                                 */
		//Pr("%d", (Int)PrIntD[i], 0);
		SyFmtPrint(stream, "%d", (Int)PrIntD[i]);

		while (i > 0)
		{
			//Pr(PRINT_FORMAT, (Int)PrIntD[--i], 0);
			SyFmtPrint(stream, "%09d", (Int)PrIntD[--i]);
		}

		//**INDENT** Pr("%<", 0, 0);


	}
	else 
	{
		//Pr("<<an integer too large to be printed>>", 0, 0);
		SyFmtPrint(stream, "<<an integer too large to be printed>>");
	}
}

/****************************************************************************
**
*F  HexStringInt( <self>, <int> ) . . . . . . . . . .  hex string for integer
*F  IntHexString( <self>, <string> ) . . . . . . . .  integer from hex string
**
**  The  function  `FuncHexStringInt'  constructs from  an  integer  the
**  corresponding string in  hexadecimal notation. It has  a leading '-'
**  for negative numbers and the digits 10..15 are written as A..F.
**
**  The  function `FuncIntHexString'  does  the converse,  but here  the
**  letters a..f are also allowed in <string> instead of A..F.
**
*/
Obj		HexStringInt(Obj integer)
{
	Int			 len;
	Int			 i;
	Int			 j;
	Int			 n;
	Int1		*p;
	TypDigit	 d;
	TypDigit	 f;
	UInt		 nf;
	UInt1		 a;
	Obj			 res;

	/* immediate integers */
	if (IS_INTOBJ(integer))
	{
		n = INT_INTOBJ(integer);

		/* 0 is special */
		if (n == 0)
		{
			res = NEW_STRING(1);
			CHARS_STRING(res)[0] = '0';
			return res;
		}

		/* else we create a string big enough for any immediate integer */
		res = NEW_STRING(2 * NR_HEX_DIGITS + 1);
		p = CHARS_STRING(res);

		/* handle sign */
		if (n < 0)
		{
			p[0] = '-';
			n = -n;
			p++;
		}
		else
		{
			SET_LEN_STRING(res, GET_LEN_STRING(res) - 1);
		}

		/* collect digits, skipping leading zeros */
		j = 0;
		nf = ((UInt)15) << (4 * (2 * NR_HEX_DIGITS - 1));

		for (i = 2 * NR_HEX_DIGITS; i; i--)
		{
			a = ((UInt)n & nf) >> (4 * (i - 1));

			if (j == 0 && a == 0) 
			{ 
				SET_LEN_STRING(res, GET_LEN_STRING(res) - 1);
			}
			else if (a < 10)
			{
				p[j++] = a + '0'; 
			}
			else 
			{
				p[j++] = a - 10 + 'A'; 
			}

			nf = nf >> 4;
		}
		/* 
		
			copy the string, so that the length is set correctly,
	  	    this is needed because in GAP3 length of string is determined by physical
	 	    length of the bag 
	   
			{} - control variable scope
	   */
		{
			Obj hdRes;
			p[j] = 0;
			C_NEW_STRING(hdRes, p);
			return hdRes;
		}
		return res;
	}
	else if (TNUM_OBJ(integer) == T_INTNEG || TNUM_OBJ(integer) == T_INTPOS)
	{
		/* nr of digits */
		len = SIZE_INT(integer);
		for (; ADDR_INT(integer)[len - 1] == 0; len--);

		/* result string and sign */
		if (TNUM_OBJ(integer) == T_INTNEG)
		{
			res = NEW_STRING(len * NR_HEX_DIGITS + 1);
			p = CHARS_STRING(res);
			p[0] = '-';
			p++;
		}
		else 
		{
			res = NEW_STRING(len * NR_HEX_DIGITS);
			p = CHARS_STRING(res);
		}

		/* collect digits */
		j = 0;
		for (; len; len--)
		{
			d = ADDR_INT(integer)[len - 1];
			f = 15L << (4 * (NR_HEX_DIGITS - 1));

			for (i = NR_HEX_DIGITS; i; i--) 
			{
				a = (d & f) >> (4 * (i - 1));

				if (j == 0 && a == 0) 
				{ 
					SET_LEN_STRING(res, GET_LEN_STRING(res) - 1); 
				}
				else if (a < 10)
				{ 
					p[j++] = a + '0'; 
				}
				else 
				{ 
					p[j++] = a - 10 + 'A'; 
				}

				f = f >> 4;
			}
		}
		/* copy the string, so that the length is set correctly,
	   this is needed because in GAP3 length of string is determined by physical
	   length of the bag */
		{
			Obj hdRes;
			p[j] = 0;
			C_NEW_STRING(hdRes, p);
			return hdRes;
		}
	}
	else
	{
		ErrorReturnObj("HexStringInt: argument must be integer, (not a %s)", (Int)TNAM_OBJ(integer), 0, "");
	}

	return (Obj)NUM_TO_UINT(0); /* please picky cc */
}

Obj		IntHexString(Obj str)
{
	Obj		 res;
	Int		 i;
	Int		 j;
	Int		 s;
	Int		 ii;
	Int		 nd;		/* number of Digits */
	Int		 len;		/* number of hex digits and sign */
	Int		 sign;
	Int1	*p;			/* collect digits, easiest to start from the end */
	UInt	 n;
	UInt1	 a;
	TypDigit d;

	if (!IS_STRING(str))
	{
		ErrorReturnObj("IntHexString: argument must be string (not a %s)", (Int)TNAM_OBJ(str), 0, "");
	}

	/* number of hex digits and sign */
	len = GET_LEN_STRING(str);

	if (len == 0) 
	{
		res = INT_INTOBJ(0);
		return res;
	}

	if (*(CHARS_STRING(str)) == '-') 
	{
		sign = -1;
		i = 1;
	}
	else 
	{
		sign = 1;
		i = 0;
	}

	/* small int case */
	if ((len - i) * 4 <= NR_SMALL_INT_BITS)
	{
		n = 0;
		p = CHARS_STRING(str);

		for (; i < len; i++) 
		{
			a = p[i];

			if ((a == 'x') || (a == 'X'))
			{
				continue;
			}

			if (a > 96)
			{
				a -= 87;
			}
			else if (a > 64)
			{
				a -= 55;
			}
			else
			{
				a -= 48;
			}

			if (a > 15)
			{
				ErrorReturnObj("IntHexString: non-valid character in hex-string", 0, 0, "");
			}

			n = (n << 4) + a;

		}

		res = INTOBJ_INT(sign * n);

		return res;
	}
	else 
	{
		/* number of Digits */
		nd = (len - i) / NR_HEX_DIGITS;

		if (nd * NR_HEX_DIGITS < (len - i)) 
		{ 
			nd++; 
		}

		nd += ((3 * nd) % 4);

		if (sign == 1)
		{
			res = NewBag(T_INTPOS, nd * sizeof(TypDigit));
		}
		else
		{
			res = NewBag(T_INTNEG, nd * sizeof(TypDigit));
		}

		/* collect digits, easiest to start from the end */
		p = CHARS_STRING(str);

		for (j = 0; j < nd; j++) 
		{
			d = 0;

			for ((s = 0), (ii = (len - j * NR_HEX_DIGITS - 1)); (ii >= i) && (ii > (len - (j + 1) * NR_HEX_DIGITS - 1)); (s += 4), ii--) 
			{
				a = p[ii];
				if ((a == 'x') || (a == 'X'))
				{
					continue;
				}

				if (a > 96)
				{
					a -= 87;
				}
				else if (a > 64)
				{
					a -= 55;
				}
				else
				{
					a -= 48;
				}

				if (a > 15)
				{
					ErrorReturnObj("IntHexString: non-valid character in hex-string", 0, 0, "");
				}

				d += (a << s);
			}

			ADDR_INT(res)[j] = d;
		}

		return res;
	}
}

/****************************************************************************
**
*F  Log2Int( <self>, <int> ) . . . . . . . . . . .  nr of bits of integer - 1
**
**  Given to GAP-Level as "Log2Int".
*/
Obj		Log2Int(Obj integer)
{
	Int			a;
	Int			len;
	Int			mask;
	Int			res;
	TypDigit	dmask;

	/* case of small ints */
	if (IS_INTOBJ(integer)) 
	{
		a = INT_INTOBJ(integer);

		if (a < 0) 
		{
			a = -a;
		}

		res = NR_SMALL_INT_BITS;

		for ((res = NR_SMALL_INT_BITS - 1), (mask = ((Int)1 << (NR_SMALL_INT_BITS - 1)));
			((mask & a) == 0 && (mask != (Int)0));
			mask = (mask >> 1), (res--));


		return INTOBJ_INT(res);
	}

	/* case of long ints */
	if (TNUM_OBJ(integer) == T_INTNEG || TNUM_OBJ(integer) == T_INTPOS) 
	{
		for (len = SIZE_INT(integer); ADDR_INT(integer)[len - 1] == 0; len--);

		res = len * NR_DIGIT_BITS - 1;
		a = (TypDigit)(ADDR_INT(integer)[len - 1]);

		for (dmask = (TypDigit)1 << (NR_DIGIT_BITS - 1);
			(dmask & a) == 0 && dmask != (TypDigit)0;
			dmask = dmask >> 1, res--);

		return INTOBJ_INT(res);
	}
	else
	{
		Error("Log2Int: argument must be integer, (not a %s)", (Int)TNAM_OBJ(integer), 0);

		return (Obj)NUM_TO_UINT(0); /* please picky cc */
	}
}

/****************************************************************************
**
*F  StringInt( <self>, <int> ) . . . . . . . . convert an integer to a string
**
**  `StringInt' returns an immutable string representing the integer <int>
**
*/
Obj		StringInt(Obj integer)
{
	Char	c;
	Int		i;
	Int		j;
	Int		x;
	Int		len;
	Int		neg;
	Int		top;
	Int		chunk;
	Obj		str;

	/* handle a small integer                                               */
	if (IS_INTOBJ(integer))
	{
		x = INT_INTOBJ(integer);
		str = NEW_STRING((NR_SMALL_INT_BITS + 5) / 3);

		/* GAP3 does not have 'IMMUTABLE' */
		/* Retype(str, T_STRING+IMMUTABLE);*/
		len = 0;

		/* Case of zero */
		if (x == 0)
		{
			CHARS_STRING(str)[0] = '0';
			CHARS_STRING(str)[1] = '\0';
			Resize(str, SIZEBAG_STRINGLEN(1));
			SET_LEN_STRING(str, 1);

			return str;
		}
		/* Negative numbers */
		if (x < 0)
		{
			CHARS_STRING(str)[len++] = '-';
			x = -x;
			neg = 1;
		}
		else
		{
			neg = 0;
		}

		/* Now the main case */
		while (x != 0)
		{
			CHARS_STRING(str)[len++] = '0' + x % 10;
			x /= 10;
		}

		CHARS_STRING(str)[len] = '\0';

		/* finally, reverse the digits in place */
		for (i = neg; i < (neg + len) / 2; i++)
		{
			c = CHARS_STRING(str)[neg + len - 1 - i];
			CHARS_STRING(str)[neg + len - 1 - i] = CHARS_STRING(str)[i];
			CHARS_STRING(str)[i] = c;
		}

		Resize(str, SIZEBAG_STRINGLEN(len));
		SET_LEN_STRING(str, len);
		return str;
	}
	/* handle a large integer                                               */
	else /*if ( SIZE_INT(integer) < 1000 )*/
	{

		/* convert the integer into base PRINT_BASE                        */
		len = IntToPrintBase(integer);
		str = NEW_STRING(CHARS_PER_PRINT_BASE * (len + 1) + 2);
		/* GAP3 does not have 'IMMUTABLE' */
		/* Retype(str, T_STRING+IMMUTABLE);*/

		/* sort out the length of the top group */
		j = 1;
		top = (Int)PrIntD[len];

		while (top >= j)
		{
			j *= 10;
		}

		/* Start filling in the string */
		i = 0;

		if (TNUM_OBJ(integer) == T_INTNEG)
		{
			CHARS_STRING(str)[i++] = '-';
		}

		while (j > 1)
		{
			j /= 10;
			CHARS_STRING(str)[i++] = '0' + (top / j) % 10;
		}

		/* Now the rest of the base PRINT_BASE digits are easy */
		while (len > 0)
		{
			chunk = (Int)PrIntD[--len];
			j = PRINT_BASE / 10;
			while (j > 0)

			{
				CHARS_STRING(str)[i++] = '0' + (chunk / j) % 10;
				j /= 10;
			}
		}

		CHARS_STRING(str)[i] = '\0';
		Resize(str, SIZEBAG_STRINGLEN(i));
		SET_LEN_STRING(str, i);

		return str;
	}
	/*
	else {
		//Very large integer, fall back on the GAP function
		return CALL_1ARGS( STRING_INT_DEFAULT, integer);
	}
	*/
}


/****************************************************************************
**
*F  EqInt( <intL>, <intR> ) . . . . . . . . .  test if two integers are equal
**
**  'EqInt' returns true if the two integer  arguments <intL> and  <intR> are
**  equal and false otherwise.
*/
Obj		EqInt(Obj	opL, Obj	opR)
{
	Int          k;              /* loop counter                    */
	TypDigit	*l;              /* pointer into the left operand   */
	TypDigit	*r;              /* pointer into the right operand  */

	/* compare two small integers                                          */
	if (ARE_INTOBJS(opL, opR)) 
	{
		if (INT_INTOBJ(opL) == INT_INTOBJ(opR))
		{
			return HdTrue; 
		}
		else
		{ 
			return HdFalse; 
		}
	}
	/* compare a small and a large integer                                 */
	else if (IS_INTOBJ(opL))
	{
		return HdFalse;
	}
	else if (IS_INTOBJ(opR)) 
	{
		return HdFalse;
	}
	/* compare two large integers                                          */
	else
	{

		/* compare the sign and size                                       */
		if (TNUM_OBJ(opL) != TNUM_OBJ(opR) || SIZE_INT(opL) != SIZE_INT(opR))
		{
			return HdFalse;
		}

		/* set up the pointers                                             */
		l = ADDR_INT(opL);
		r = ADDR_INT(opR);

		/* run through the digits, four at a time                          */
		for (k = SIZE_INT(opL) / 4 - 1; k >= 0; k--) 
		{
			if (*l++ != *r++) 
			{ 
				return HdFalse;
			}
			if (*l++ != *r++) 
			{ 
				return HdFalse;
			}
			if (*l++ != *r++)
			{ 
				return HdFalse;
			}
			if (*l++ != *r++) 
			{ 
				return HdFalse; 
			}
		}

		/* no differences found, so they must be equal                     */
		return HdTrue;

	}
}


/****************************************************************************
**
*F  LtInt( <intL>, <intR> ) . . . . . test if an integer is less than another
**
**  'LtInt' returns true if the  integer  <intL>  is  strictly  less than the
**  integer
**  <intR> and false otherwise.
*/
Obj		LtInt(Obj	opL, Obj	opR)
{
	Int          k;              /* loop counter                    */
	TypDigit	*l;              /* pointer into the left operand   */
	TypDigit	*r;              /* pointer into the right operand  */

	/* compare two small integers                                          */
	if (ARE_INTOBJS(opL, opR))
	{
		if (INT_INTOBJ(opL) < INT_INTOBJ(opR)) 
		{ 
			return HdTrue; 
		}
		else 
		{
			return HdFalse;
		}
	}

	/* compare a small and a large integer                                 */
	else if (IS_INTOBJ(opL))
	{
		if (TNUM_OBJ(opR) == T_INTPOS)
		{ 
			return HdTrue;
		}
		else 
		{ 
			return HdFalse; 
		}
	}
	else if (IS_INTOBJ(opR)) 
	{
		if (TNUM_OBJ(opL) == T_INTPOS)
		{ 
			return HdFalse;
		}
		else 
		{ 
			return HdTrue;
		}
	}
	/* compare two large integers                                          */
	else {

		/* compare the sign and size                                       */
		if (TNUM_OBJ(opL) == T_INTNEG && TNUM_OBJ(opR) == T_INTPOS)
		{
			return HdTrue;
		}
		else if (TNUM_OBJ(opL) == T_INTPOS && TNUM_OBJ(opR) == T_INTNEG)
		{
			return HdFalse;
		}
		else if ((TNUM_OBJ(opL) == T_INTPOS && SIZE_INT(opL) < SIZE_INT(opR)) || (TNUM_OBJ(opL) == T_INTNEG && SIZE_INT(opL) > SIZE_INT(opR)))
		{
			return HdTrue;
		}
		else if ((TNUM_OBJ(opL) == T_INTPOS && SIZE_INT(opL) > SIZE_INT(opR)) || (TNUM_OBJ(opL) == T_INTNEG && SIZE_INT(opL) < SIZE_INT(opR)))
		{
			return HdFalse;
		}

		/* set up the pointers                                             */
		l = ADDR_INT(opL);
		r = ADDR_INT(opR);

		/* run through the digits, from the end downwards                  */
		for (k = SIZE_INT(opL) - 1; k >= 0; k--) 
		{
			if (l[k] != r[k]) 
			{
				if ((TNUM_OBJ(opL) == T_INTPOS && l[k] < r[k]) || (TNUM_OBJ(opL) == T_INTNEG && l[k] > r[k]))
				{
					return HdTrue;
				}
				else
				{
					return HdFalse;
				}
			}
		}

		/* no differences found, so they must be equal                     */
		return HdFalse;

	}
}


/****************************************************************************
**
*F  SumInt( <intL>, <intR> )  . . . . . . . . . . . . . . sum of two integers
**
**  'SumInt' returns the sum of the two integer arguments <intL> and  <intR>.
**  'SumInt' handles operands of type 'T_INT', 'T_INTPOS' and 'T_INTNEG'.
**
**  It can also be used in the cases that both operands  are  small  integers
**  and the result is a small integer too,  i.e., that  no  overflow  occurs.
**  This case is usually already handled in 'EvSum' for a better  efficiency.
**
**  Is called from the 'EvSum'  binop so both operands are already evaluated.
**
**  'SumInt' is a little bit difficult since there are 16  different cases to
**  handle, each operand can be positive or negative, small or large integer.
**  If the operands have opposite sign 'SumInt' calls 'DiffInt',  this  helps
**  reduce the total amount of code by a factor of two.
*/
Obj		SumInt(Obj opL, Obj opR)
{
	Int          i;              /* loop variable                   */
	Int          k;              /* loop variable                   */
	Int          cs;             /* sum of two smalls */
	TypDigit	*l;              /* pointer into the left operand   */
	TypDigit	*r;              /* pointer into the right operand  */
	TypDigit	*s;              /* pointer into the sum            */
	UInt         c;              /* sum of two digits               */
	UInt		*l2;             /* pointer to get 2 digits at once */
	UInt		*s2;             /* pointer to put 2 digits at once */
	Obj          sum;            /* handle of the result bag        */

	/* adding two small integers                                           */
	if (ARE_INTOBJS(opL, opR))
	{

		/* add two small integers with a small sum                         */
		/* add and compare top two bits to check that no overflow occured  */
		if (SUM_INTOBJS(sum, opL, opR)) 
		{
			return sum;
		}

		/* add two small integers with a large sum                         */
		cs = INT_INTOBJ(opL) + INT_INTOBJ(opR);

		if (0 < cs)
		{
			sum = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
			ADDR_INT(sum)[0] = (TypDigit)cs;
			ADDR_INT(sum)[1] = (TypDigit)(cs >> NR_DIGIT_BITS);
		}
		else 
		{
			sum = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
			ADDR_INT(sum)[0] = (TypDigit)(-cs);
			ADDR_INT(sum)[1] = (TypDigit)((-cs) >> NR_DIGIT_BITS);
		}
	}
	/* adding one large integer and one small integer                      */
	else if (IS_INTOBJ(opL) || IS_INTOBJ(opR)) 
	{
		/* make the right operand the small one                            */
		if (IS_INTOBJ(opL)) 
		{
			sum = opL;
			opL = opR;
			opR = sum;
		}

		/* if the integers have different sign, let 'DiffInt' do the work  */
		if ((TNUM_OBJ(opL) == T_INTNEG && 0 <= INT_INTOBJ(opR)) || (TNUM_OBJ(opL) == T_INTPOS && INT_INTOBJ(opR) < 0)) 
		{
			if (TNUM_OBJ(opL) == T_INTPOS)
			{ 
				Retype(opL, T_INTNEG);
			}
			else 
			{
				Retype(opL, T_INTPOS);
			}

			sum = DiffInt(opR, opL);

			if (TNUM_OBJ(opL) == T_INTPOS)
			{ 
				Retype(opL, T_INTNEG);
			}
			else 
			{
				Retype(opL, T_INTPOS); 
			}

			return sum;
		}

		/* allocate the result bag and set up the pointers                 */
		if (TNUM_OBJ(opL) == T_INTPOS) 
		{
			i = INT_INTOBJ(opR);
			sum = NewBag(T_INTPOS, (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		}
		else 
		{
			i = -INT_INTOBJ(opR);
			sum = NewBag(T_INTNEG, (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		}

		l = ADDR_INT(opL);
		s = ADDR_INT(sum);

		/* add the first four digits,the right operand has only two digits */
		c = (UInt)* l++ + (TypDigit)i;                             
		*s++ = (TypDigit)c;
		c = (UInt)* l++ + (i >> NR_DIGIT_BITS) + (c >> NR_DIGIT_BITS);
		*s++ = (TypDigit)c;
		c = (UInt)* l++ + (c >> NR_DIGIT_BITS);
		*s++ = (TypDigit)c;
		c = (UInt)* l++ + (c >> NR_DIGIT_BITS);
		*s++ = (TypDigit)c;

		/* propagate the carry, this loop is almost never executed         */
		for (k = SIZE_INT(opL) / 4 - 1; k != 0 && (c >> NR_DIGIT_BITS) != 0; k--)
		{
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);
			*s++ = (TypDigit)c;
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);
			*s++ = (TypDigit)c;
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);
			*s++ = (TypDigit)c;
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);
			*s++ = (TypDigit)c;
		}

		/* just copy the remaining digits, do it two digits at once        */
		for (l2 = (UInt*)l, s2 = (UInt*)s; k != 0; k--) 
		{
			*s2++ = *l2++;
			*s2++ = *l2++;
		}

		/* if there is a carry, enter it, otherwise shrink the sum         */
		if ((c >> NR_DIGIT_BITS) != 0)
		{
			*s++ = (TypDigit)(c >> NR_DIGIT_BITS);
		}
		else
		{
			Resize(sum, (SIZE_INT(sum) - 4) * sizeof(TypDigit));
		}

	}
	/* add two large integers                                              */
	else
 {

		/* if the integers have different sign, let 'DiffInt' do the work  */
		if ((TNUM_OBJ(opL) == T_INTPOS && TNUM_OBJ(opR) == T_INTNEG) || (TNUM_OBJ(opL) == T_INTNEG && TNUM_OBJ(opR) == T_INTPOS)) 
		 {
			if (TNUM_OBJ(opL) == T_INTPOS) 
			{ 
				Retype(opL, T_INTNEG);
			}
			else 
			{
				Retype(opL, T_INTPOS); 
			}

			sum = DiffInt(opR, opL);

			if (TNUM_OBJ(opL) == T_INTPOS)
			{ 
				Retype(opL, T_INTNEG);
			}
			else
			{ 
				Retype(opL, T_INTPOS); 
			}

			return sum;
		}

		/* make the right operand the smaller one                          */
		if (SIZE_INT(opL) < SIZE_INT(opR))
		{
			sum = opL;  opL = opR;  opR = sum;
		}

		/* allocate the result bag and set up the pointers                 */
		if (TNUM_OBJ(opL) == T_INTPOS) 
		{
			sum = NewBag(T_INTPOS, (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		}
		else 
		{
			sum = NewBag(T_INTNEG, (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		}

		l = ADDR_INT(opL);
		r = ADDR_INT(opR);
		s = ADDR_INT(sum);

		/* add the digits, convert to UInt to get maximum precision         */
		c = 0;

		for (k = SIZE_INT(opR) / 4; k != 0; k--) 
		{
			c = (UInt)* l++ + (UInt)* r++ + (c >> NR_DIGIT_BITS);  *s++ = (TypDigit)c;
			c = (UInt)* l++ + (UInt)* r++ + (c >> NR_DIGIT_BITS);  *s++ = (TypDigit)c;
			c = (UInt)* l++ + (UInt)* r++ + (c >> NR_DIGIT_BITS);  *s++ = (TypDigit)c;
			c = (UInt)* l++ + (UInt)* r++ + (c >> NR_DIGIT_BITS);  *s++ = (TypDigit)c;
		}

		/* propagate the carry, this loop is almost never executed         */
		for (k = (SIZE_INT(opL) - SIZE_INT(opR)) / 4; k != 0 && (c >> NR_DIGIT_BITS) != 0; k--)
		{
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);  
			*s++ = (TypDigit)c;
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);  
			*s++ = (TypDigit)c;
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);  
			*s++ = (TypDigit)c;
			c = (UInt)* l++ + (c >> NR_DIGIT_BITS);  
			*s++ = (TypDigit)c;
		}

		/* just copy the remaining digits, do it two digits at once        */
		for (l2 = (UInt*)l, s2 = (UInt*)s; k != 0; k--)
		{
			*s2++ = *l2++;
			*s2++ = *l2++;
		}

		/* if there is a carry, enter it, otherwise shrink the sum         */
		if ((c >> NR_DIGIT_BITS) != 0)
		{
			*s++ = (TypDigit)(c >> NR_DIGIT_BITS);
		}
		else
		{
			Resize(sum, (SIZE_INT(sum) - 4) * sizeof(TypDigit));
		}

	}

	/* return the sum                                                      */
	return sum;
}


/****************************************************************************
**
*F  AInvInt(<int>)  . . . . . . . . . . . . .  additive inverse of an integer
*/
Obj		AInvInt(Obj op)
{
	Obj		inv;
	UInt	i;

	/* handle small integer                                                */
	if (IS_INTOBJ(op)) 
	{
		/* special case (ugh)                                              */
		if (op == INTOBJ_INT(-(NUM_TO_INT(1) << NR_SMALL_INT_BITS))) 
		{
			inv = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
			ADDR_INT(inv)[0] = 0;
			ADDR_INT(inv)[1] = (TypDigit)(NUM_TO_INT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS));
		}
		/* general case                                                    */
		else
		{
			inv = INTOBJ_INT(-INT_INTOBJ(op));
		}

	}
	/* invert a large integer                                              */
	else
	{

		/* special case (ugh)                                              */
		if ((TNUM_OBJ(op) == T_INTPOS) && (SIZE_INT(op) == 4) && (ADDR_INT(op)[3] == 0) && (ADDR_INT(op)[2] == 0)
			&& (ADDR_INT(op)[1] == (NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS))) && (ADDR_INT(op)[0] == 0)) 
		{
			inv = INTOBJ_INT(-(NUM_TO_INT(1) << NR_SMALL_INT_BITS));
		}
		/* general case                                                    */
		else
		{
			if (TNUM_OBJ(op) == T_INTPOS)
			{
				inv = NewBag(T_INTNEG, SIZE_OBJ(op));
			}
			else 
			{
				inv = NewBag(T_INTPOS, SIZE_OBJ(op));
			}

			for (i = 0; i < SIZE_INT(op); i++) 
			{
				ADDR_INT(inv)[i] = ADDR_INT(op)[i];
			}
		}

	}

	/* return the inverse                                                  */
	return inv;
}


/****************************************************************************
**
*F  DiffInt( <intL>, <intR> ) . . . . . . . . . .  difference of two integers
**
**  'DiffInt' returns the difference of the two integer arguments <intL>  and
**  <intR>.  'DiffInt' handles  operands  of  type  'T_INT',  'T_INTPOS'  and
**  'T_INTNEG'.
**
**  It can also be used in the cases that both operands  are  small  integers
**  and the result is a small integer too,  i.e., that  no  overflow  occurs.
**  This case is usually already handled in 'EvDiff' for a better efficiency.
**
**  Is called from the 'EvDiff' binop so both operands are already evaluated.
**
**  'DiffInt' is a little bit difficult since there are 16 different cases to
**  handle, each operand can be positive or negative, small or large integer.
**  If the operands have opposite sign 'DiffInt' calls 'SumInt',  this  helps
**  reduce the total amount of code by a factor of two.
*/
Obj		DiffInt(Obj opL,Obj opR)
{
	Int          i;              /* loop variable                   */
	Int          k;              /* loop variable                   */
	Int          c;              /* difference of two digits        */
	TypDigit	*d;              /* pointer into the difference     */
	TypDigit	*l;              /* pointer into the left operand   */
	TypDigit	*r;              /* pointer into the right operand  */
	UInt		*d2;             /* pointer to put 2 digits at once */
	UInt		*l2;             /* pointer to get 2 digits at once */
	Obj          dif;            /* handle of the result bag        */

	/* subtracting two small integers                                      */
	if (ARE_INTOBJS(opL, opR)) 
	{

		/* subtract two small integers with a small difference             */
		/* sub and compare top two bits to check that no overflow occured  */
		if (DIFF_INTOBJS(dif, opL, opR)) 
		{
			return dif;
		}

		/* subtract two small integers with a large difference             */
		c = INT_INTOBJ(opL) - INT_INTOBJ(opR);

		if (0 < c)
		{
			dif = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
			ADDR_INT(dif)[0] = (TypDigit)c;
			ADDR_INT(dif)[1] = (TypDigit)(c >> NR_DIGIT_BITS);
		}
		else 
		{
			dif = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
			ADDR_INT(dif)[0] = (TypDigit)(-c);
			ADDR_INT(dif)[1] = (TypDigit)((-c) >> NR_DIGIT_BITS);
		}
	}
	/* subtracting one small integer and one large integer                 */
	else if (IS_INTOBJ(opL) || IS_INTOBJ(opR)) 
	{
		/* make the right operand the small one                            */
		if (IS_INTOBJ(opL))
		{
			dif = opL;  opL = opR;  opR = dif;
			c = -1;
		}
		else
		{
			c = 1;
		}

		/* if the integers have different sign, let 'SumInt' do the work   */
		if ((TNUM_OBJ(opL) == T_INTNEG && 0 <= INT_INTOBJ(opR)) || (TNUM_OBJ(opL) == T_INTPOS && INT_INTOBJ(opR) < 0))
		{

			if (TNUM_OBJ(opL) == T_INTPOS) 
			{
				Retype(opL, T_INTNEG); 
			}
			else
			{ 
				Retype(opL, T_INTPOS); 
			}

			dif = SumInt(opL, opR);

			if (TNUM_OBJ(opL) == T_INTPOS) 
			{ 
				Retype(opL, T_INTNEG);
			}
			else 
			{ 
				Retype(opL, T_INTPOS); 
			}

			if (c == 1) 
			{
				if (TNUM_OBJ(dif) == T_INTPOS) 
				{
					Retype(dif, T_INTNEG);
				}
				else 
				{ 
					Retype(dif, T_INTPOS); 
				}
			}

			return dif;
		}

		/* allocate the result bag and set up the pointers                 */
		if (TNUM_OBJ(opL) == T_INTPOS)
		{
			i = INT_INTOBJ(opR);
			if (c == 1) 
			{
				dif = NewBag(T_INTPOS, SIZE_OBJ(opL));
			}
			else 
			{ 
				dif = NewBag(T_INTNEG, SIZE_OBJ(opL));
			}
		}
		else 
		{
			i = -INT_INTOBJ(opR);
			if (c == 1)
			{
				dif = NewBag(T_INTNEG, SIZE_OBJ(opL));
			}
			else 
			{ 
				dif = NewBag(T_INTPOS, SIZE_OBJ(opL));
			}
		}

		l = ADDR_INT(opL);
		d = ADDR_INT(dif);

		/* sub the first four digit, note the left operand has only two    */
		/*N (c>>16<) need not work, replace by (c<0?-1:0)                   */
		c = (Int)* l++ - (TypDigit)i;
		*d++ = (TypDigit)c;
		c = (Int)* l++ - (TypDigit)(i / (NUM_TO_UINT(1) << NR_DIGIT_BITS)) + (c < 0 ? -1 : 0); 
		*d++ = (TypDigit)c;
		c = (Int)* l++ + (c < 0 ? -1 : 0);  
		*d++ = (TypDigit)c;
		c = (Int)* l++ + (c < 0 ? -1 : 0);  
		*d++ = (TypDigit)c;

		/* propagate the carry, this loop is almost never executed         */
		for (k = SIZE_INT(opL) / 4 - 1; k != 0 && c < 0; k--)
		{
			c = (Int)* l++ + (c < 0 ? -1 : 0);  
			*d++ = (TypDigit)c;
			c = (Int)* l++ + (c < 0 ? -1 : 0); 
			*d++ = (TypDigit)c;
			c = (Int)* l++ + (c < 0 ? -1 : 0);  
			*d++ = (TypDigit)c;
			c = (Int)* l++ + (c < 0 ? -1 : 0);  
			*d++ = (TypDigit)c;
		}

		/* just copy the remaining digits, do it two digits at once        */
		for (l2 = (UInt*)l, d2 = (UInt*)d; k != 0; k--) 
		{
			*d2++ = *l2++;
			*d2++ = *l2++;
		}

		/* no underflow since we subtracted a small int from a large one   */
		/* but there may be leading zeroes in the result, get rid of them  */
		/* occurs almost never, so it doesn't matter that it is expensive  */
		if (((UInt*)d == d2 && d[-4] == 0 && d[-3] == 0 && d[-2] == 0 && d[-1] == 0) || (SIZE_INT(dif) == 4 && d[-2] == 0 && d[-1] == 0))
		{
			/* find the number of significant digits                       */
			d = ADDR_INT(dif);

			for (k = SIZE_INT(dif); k != 0; k--) 
			{
				if (d[k - 1] != 0)
				{
					break;
				}
			}

			/* reduce to small integer if possible, otherwise shrink bag   */
			if (k <= 2 && TNUM_OBJ(dif) == T_INTPOS && (UInt)(INTBASE * d[1] + d[0]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				dif = INTOBJ_INT(INTBASE * d[1] + d[0]);
			}
			else if (k <= 2 && TNUM_OBJ(dif) == T_INTNEG && (UInt)(INTBASE * d[1] + d[0]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				dif = INTOBJ_INT(-(Int)(INTBASE * d[1] + d[0]));
			}
			else
			{
				Resize(dif, (((k + 3) / 4) * 4) * sizeof(TypDigit));
			}
		}

	}
	/* subtracting two large integers                                      */
	else
 {

		/* if the integers have different sign, let 'SumInt' do the work   */
		if ((TNUM_OBJ(opL) == T_INTPOS && TNUM_OBJ(opR) == T_INTNEG) || (TNUM_OBJ(opL) == T_INTNEG && TNUM_OBJ(opR) == T_INTPOS)) 
		{
			if (TNUM_OBJ(opR) == T_INTPOS) 
			{ 
				Retype(opR, T_INTNEG); 
			}
			else 
			{
				Retype(opR, T_INTPOS);
			}

			dif = SumInt(opL, opR);

			if (TNUM_OBJ(opR) == T_INTPOS)
			{ 
				Retype(opR, T_INTNEG);
			}
			else 
			{ 
				Retype(opR, T_INTPOS);
			}

			return dif;
		}

		/* make the right operand the smaller one                          */
		if (SIZE_INT(opL) < SIZE_INT(opR) || (TNUM_OBJ(opL) == T_INTPOS && LtInt(opL, opR) == HdTrue) || (TNUM_OBJ(opL) == T_INTNEG && LtInt(opR, opL) == HdTrue))
		{
			dif = opL; 
			opL = opR; 
			opR = dif;
			c = -1;
		}
		else
		{
			c = 1;
		}

		/* allocate the result bag and set up the pointers                 */
		if ((TNUM_OBJ(opL) == T_INTPOS && c == 1) || (TNUM_OBJ(opL) == T_INTNEG && c == -1))
		{
			dif = NewBag(T_INTPOS, SIZE_OBJ(opL));
		}
		else
		{
			dif = NewBag(T_INTNEG, SIZE_OBJ(opL));
		}

		l = ADDR_INT(opL);
		r = ADDR_INT(opR);
		d = ADDR_INT(dif);

		/* subtract the digits                                             */
		c = 0;

		for (k = SIZE_INT(opR) / 4; k != 0; k--)
		{
			c = (Int)* l++ - (Int)* r++ + (c < 0 ? -1 : 0); 
			*d++ = (TypDigit)c;
			c = (Int)* l++ - (Int)* r++ + (c < 0 ? -1 : 0);
			*d++ = (TypDigit)c;
			c = (Int)* l++ - (Int)* r++ + (c < 0 ? -1 : 0); 
			*d++ = (TypDigit)c;
			c = (Int)* l++ - (Int)* r++ + (c < 0 ? -1 : 0); 
			*d++ = (TypDigit)c;
		}

		/* propagate the carry, this loop is almost never executed         */
		for (k = (SIZE_INT(opL) - SIZE_INT(opR)) / 4; k != 0 && c < 0; k--) 
		{
			c = (Int)* l++ + (c < 0 ? -1 : 0);
			*d++ = (TypDigit)c;
			c = (Int)* l++ + (c < 0 ? -1 : 0);
			*d++ = (TypDigit)c;
			c = (Int)* l++ + (c < 0 ? -1 : 0); 
			*d++ = (TypDigit)c;
			c = (Int)* l++ + (c < 0 ? -1 : 0);
			*d++ = (TypDigit)c;
		}

		/* just copy the remaining digits, do it two digits at once        */
		for (d2 = (UInt*)d, l2 = (UInt*)l; k != 0; k--) 
		{
			*d2++ = *l2++;
			*d2++ = *l2++;
		}

		/* no underflow since we subtracted a small int from a large one   */
		/* but there may be leading zeroes in the result, get rid of them  */
		/* occurs almost never, so it doesn't matter that it is expensive  */
		if (((UInt*)d == d2 && d[-4] == 0 && d[-3] == 0 && d[-2] == 0 && d[-1] == 0) || (SIZE_INT(dif) == 4 && d[-2] == 0 && d[-1] == 0))
		{
			/* find the number of significant digits                       */
			d = ADDR_INT(dif);

			for (k = SIZE_INT(dif); k != 0; k--) 
			{
				if (d[k - 1] != 0)
				{
					break;
				}
			}

			/* reduce to small integer if possible, otherwise shrink bag   */
			if (k <= 2 && TNUM_OBJ(dif) == T_INTPOS && (UInt)(INTBASE * d[1] + d[0]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				dif = INTOBJ_INT(INTBASE * d[1] + d[0]);
			}
			else if (k <= 2 && TNUM_OBJ(dif) == T_INTNEG && (UInt)(INTBASE * d[1] + d[0]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				dif = INTOBJ_INT(-(Int)(INTBASE * d[1] + d[0]));
			}
			else
			{
				Resize(dif, (((k + 3) / 4) * 4) * sizeof(TypDigit));
			}

		}

	}

	/* return the difference                                               */
	return dif;
}


/*
 *  prod_intobjs() multiplies two immediate (i.e., small) integers and returns
 *  the product as an immediate integer; if the product overflows then zero is
 *  returned.  NOTE: a valid result of zero is returned as INTOBJ_INT(0) which
 *  is not equal to zero.
 */

static Obj prod_intobjs(Obj l, Obj r)
{
	Int left = INT_INTOBJ(l);
	Int right = INT_INTOBJ(r);
	Int result = 0;

	if (left == 0 || right == 0)
	{
		return INTOBJ_INT(0);
	}

	if (MAX_SMALL_INTEGER / llabs(left) < llabs(right))
	{
		// product of left * right will overflow to large int
		return result;
	}
	
	result = left * right;
	return INTOBJ_INT(result);
}


/****************************************************************************
**
*F  ProdInt( <intL>, <intR> ) . . . . . . . . . . . . product of two integers
**
**  'ProdInt' returns the product of the two  integer  arguments  <intL>  and
**  <intR>.  'ProdInt' handles  operands  of  type  'T_INT',  'T_INTPOS'  and
**  'T_INTNEG'.
**
**  It can also be used in the cases that both operands  are  small  integers
**  and the result is a small integer too,  i.e., that  no  overflow  occurs.
**  This case is usually already handled in 'EvProd' for a better efficiency.
**
**  Is called from the 'EvProd' binop so both operands are already evaluated.
**
**  The only difficult about this function is the fact that is has two handle
**  3 different situation, depending on how many arguments  are  small  ints.
*/
Obj		ProdInt(Obj opL, Obj opR)
{
	Int              i;              /* loop count, value for small int */
	Int              k;              /* loop count, value for small int */
	UInt             c;              /* product of two digits           */
	TypDigit         l;              /* one digit of the left operand   */
	TypDigit		*r;              /* pointer into the right operand  */
	TypDigit		*p;              /* pointer into the product        */
	Obj              prd;            /* handle of the result bag        */

	/* multiplying two small integers                                      */
	if (ARE_INTOBJS(opL, opR))
	{
		/* multiply two small integers: check result is a small product    */
		prd = prod_intobjs(opL, opR);

		if (prd != 0) 
		{
			return prd;
		}

		/* result is not a small product: get the integer values           */
		i = INT_INTOBJ(opL);
		k = INT_INTOBJ(opR);

		/* allocate the product bag                                        */
		if ((0 < i && 0 < k) || (i < 0 && k < 0))
		{
			prd = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
		}
		else
		{
			prd = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
		}

		p = ADDR_INT(prd);

		/* make both operands positive                                     */
		if (i < 0)
		{ 
			i = -i;
		}

		if (k < 0) 
		{
			k = -k; 
		}

		/* multiply digitwise                                              */
		c = (UInt)(TypDigit)i * (TypDigit)k;
		p[0] = (TypDigit)c;

		c = (UInt)(TypDigit)i * (((UInt)k) >> NR_DIGIT_BITS)+ (c >> NR_DIGIT_BITS);

		p[1] = (TypDigit)c;
		p[2] = c >> NR_DIGIT_BITS;

		c = (UInt)(TypDigit)(((UInt)i) >> NR_DIGIT_BITS) * (TypDigit)k + p[1];
		p[1] = (TypDigit)c;
		c = (UInt)(TypDigit)(((UInt)i) >> NR_DIGIT_BITS) * (TypDigit)(((UInt)k) >> NR_DIGIT_BITS) + p[2] + (c >> NR_DIGIT_BITS);
		p[2] = (TypDigit)c;
		p[3] = (TypDigit)(c >> NR_DIGIT_BITS);

	}
	/* multiply a small and a large integer                                */
	else if (IS_INTOBJ(opL) || IS_INTOBJ(opR)) 
	{

		/* make the left operand the small one                             */
		if (IS_INTOBJ(opR)) 
		{
			i = INT_INTOBJ(opR);  
			opR = opL;
		}
		else 
		{
			i = INT_INTOBJ(opL);
		}

		/* handle trivial cases first                                      */
		if (i == 0)
		{
			return INTOBJ_INT(0);
		}

		if (i == 1)
		{
			return opR;
		}

		/* the large integer 1<<28 times -1 is the small integer -(1<<28)  */
		if (i == -1
			&& TNUM_OBJ(opR) == T_INTPOS && SIZE_INT(opR) == 4
			&& ADDR_INT(opR)[3] == 0
			&& ADDR_INT(opR)[2] == 0
			&& ADDR_INT(opR)[1] == (NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS))
			&& ADDR_INT(opR)[0] == 0)
		{
			return INTOBJ_INT(-(Int)(NUM_TO_UINT(1) << NR_SMALL_INT_BITS));
		}

		/* multiplication by -1 is easy, just switch the sign and copy     */
		if (i == -1)
		{
			if (TNUM_OBJ(opR) == T_INTPOS)
			{
				prd = NewBag(T_INTNEG, SIZE_OBJ(opR));
			}
			else
			{
				prd = NewBag(T_INTPOS, SIZE_OBJ(opR));
			}

			r = ADDR_INT(opR);
			p = ADDR_INT(prd);

			for (k = SIZE_INT(opR) / 4; k != 0; k--) 
			{
				/*N should be: *p2++=*r2++;  *p2++=*r2++;                  */
				*p++ = *r++; 
				*p++ = *r++; 
				*p++ = *r++; 
				*p++ = *r++;
			}

			return prd;
		}

		/* allocate a bag for the result                                   */
		if ((0 < i && TNUM_OBJ(opR) == T_INTPOS) || (i < 0 && TNUM_OBJ(opR) == T_INTNEG))
		{
			prd = NewBag(T_INTPOS, (SIZE_INT(opR) + 4) * sizeof(TypDigit));
		}
		else
		{
			prd = NewBag(T_INTNEG, (SIZE_INT(opR) + 4) * sizeof(TypDigit));
		}

		if (i < 0) 
		{ 
			i = -i; 
		}

		/* multiply with the lower digit of the left operand               */
		l = (TypDigit)i;

		if (l != 0)
		{

			r = ADDR_INT(opR);
			p = ADDR_INT(prd);
			c = 0;

			/* multiply the right with this digit and store in the product */
			for (k = SIZE_INT(opR) / 4; k != 0; k--) 
			{
				c = (UInt)l * (UInt)* r++ + (c >> NR_DIGIT_BITS); 
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (c >> NR_DIGIT_BITS); 
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (c >> NR_DIGIT_BITS); 
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (c >> NR_DIGIT_BITS); 
				*p++ = (TypDigit)c;
			}

			*p = (TypDigit)(c >> NR_DIGIT_BITS);
		}

		/* multiply with the larger digit of the left operand              */
		l = ((UInt)i) >> NR_DIGIT_BITS;

		if (l != 0)
		{

			r = ADDR_INT(opR);
			p = ADDR_INT(prd) + 1;
			c = 0;

			/* multiply the right with this digit and add into the product */
			for (k = SIZE_INT(opR) / 4; k != 0; k--) 
			{
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
			}

			*p = (TypDigit)(c >> NR_DIGIT_BITS);
		}

		/* remove the leading zeroes, note that there can't be more than 6 */
		p = ADDR_INT(prd) + SIZE_INT(prd);

		if ((p[-4] == 0) && (p[-3] == 0) && (p[-2] == 0) && (p[-1] == 0))
		{
			Resize(prd, (SIZE_INT(prd) - 4) * sizeof(TypDigit));
		}

	}
	/* multiply two large integers                                         */
	else
	{

		/* make the left operand the smaller one, for performance          */
		if (SIZE_INT(opL) > SIZE_INT(opR)) 
		{
			prd = opR; 
			opR = opL;  
			opL = prd;
		}

		/* allocate a bag for the result                                   */
		if (TNUM_OBJ(opL) == TNUM_OBJ(opR))
		{
			prd = NewBag(T_INTPOS, SIZE_OBJ(opL) + SIZE_OBJ(opR));
		}
		else
		{
			prd = NewBag(T_INTNEG, SIZE_OBJ(opL) + SIZE_OBJ(opR));
		}

		/* run through the digits of the left operand                      */
		for (i = 0; i < (Int)SIZE_INT(opL); i++)
		{

			/* set up pointer for one loop iteration                       */
			l = ADDR_INT(opL)[i];
			if (l == 0) 
			{ 
				continue; 
			}
			r = ADDR_INT(opR);
			p = ADDR_INT(prd) + i;
			c = 0;

			/* multiply the right with this digit and add into the product */
			for (k = SIZE_INT(opR) / 4; k != 0; k--)
			{
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS); 
				*p++ = (TypDigit)c;
				c = (UInt)l * (UInt)* r++ + (UInt)* p + (c >> NR_DIGIT_BITS);
				*p++ = (TypDigit)c;
			}
			*p = (TypDigit)(c >> NR_DIGIT_BITS);
		}

		/* remove the leading zeroes, note that there can't be more than 7 */
		p = ADDR_INT(prd) + SIZE_INT(prd);

		if (p[-4] == 0 && p[-3] == 0 && p[-2] == 0 && p[-1] == 0) 
		{
			Resize(prd, (SIZE_INT(prd) - 4) * sizeof(TypDigit));
		}

	}

	/* return the product                                                  */
	return prd;
}


/****************************************************************************
 **
 *F  PowInt( <intL>, <intR> )  . . . . . . . . . . . . . . power of an integer
 **
 **  'PowInt' returns the <intR>-th (an integer) power of the integer  <intL>.
 **  'PowInt' handles operands of type 'T_INT', 'T_INTPOS' and 'T_INTNEG'.
 **
 **  It can also be used in the cases that both operands  are  small  integers
 **  and the result is a small integer too,  i.e., that  no  overflow  occurs.
 **  This case is usually already handled in 'EvPow' for a better  efficiency.
 **
 **  Is called from the 'EvPow'  binop so both operands are already evaluated.
 */
Obj		PowInt(Obj opL,	Obj opR)
{

	Int		i;
	Obj     pow;

	/* power with a large exponent                                         */
	if (!IS_INTOBJ(opR))
	{
		if (opL == INTOBJ_INT(0))
		{
			pow = INTOBJ_INT(0);
		}
		else if (opL == INTOBJ_INT(1))
		{
			pow = INTOBJ_INT(1);
		}
		else if (opL == INTOBJ_INT(-1) && ADDR_INT(opR)[0] % 2 == 0)
		{
			pow = INTOBJ_INT(1);
		}
		else if (opL == INTOBJ_INT(-1) && ADDR_INT(opR)[0] % 2 != 0)
		{
			pow = INTOBJ_INT(-1);
		}
		else 
		{
			opR = ErrorReturnObj("Integer operands: <exponent> is too large", 0, 0, "you can replace the integer <exponent> via 'return <exponent>;'");
			return POW(opL, opR);
		}
	}
	/* power with a negative exponent                                      */
	else if (INT_INTOBJ(opR) < 0)
	{

		if (opL == INTOBJ_INT(0))
		{
			opL = ErrorReturnObj("Integer operands: <base> must not be zero", 0, 0, "you can replace the integer <base> via 'return <base>;'");
			return POW(opL, opR);
		}
		else if (opL == INTOBJ_INT(1))
		{
			pow = INTOBJ_INT(1);
		}
		else if (opL == INTOBJ_INT(-1) && INT_INTOBJ(opR) % 2 == 0)
		{
			pow = INTOBJ_INT(1);
		}
		else if (opL == INTOBJ_INT(-1) && INT_INTOBJ(opR) % 2 != 0)
		{
			pow = INTOBJ_INT(-1);
		}
		else
		{
			pow = QUO(INTOBJ_INT(1), PowInt(opL, INTOBJ_INT(-INT_INTOBJ(opR))));
		}
	}
	/* power with a small positive exponent, do it by a repeated squaring  */
	else 
	{
		pow = INTOBJ_INT(1);
		i = INT_INTOBJ(opR);

		while (i != 0)
		{
			if (i % 2 == 1) 
			{
				pow = ProdInt(pow, opL);
			}

			if (i > 1)
			{
				opL = ProdInt(opL, opL);
			}

			i = i / 2;
		}
	}
	/* return the power                                                    */
	return pow;
}


/****************************************************************************
 **
 *F  ModInt( <intL>, <intR> )  . representative of residue class of an integer
 **
 **  'ModInt' returns the smallest positive representant of the residue  class
 **  of the  integer  <intL>  modulo  the  integer  <intR>.  'ModInt'  handles
 **  operands of type 'T_INT', 'T_INTPOS', 'T_INTNEG'.
 **
 **  It can also be used in the cases that both operands  are  small  integers
 **  and the result is a small integer too,  i.e., that  no  overflow  occurs.
 **  This case is usually already handled in 'EvMod' for a better efficiency.
p**
**  Is called from the 'EvMod'  binop so both operands are already evaluated.
*/
Obj		ModInt(Obj opL, Obj opR)
{
	Int              i;              /* loop count, value for small int */
	Int              k;              /* loop count, value for small int */
	TypDigit         d;              /* carry into the next digit       */
	TypDigit		*l;              /* pointer into the left operand   */
	TypDigit		*m;              /* pointer into the remainder      */
	TypDigit		*r;              /* pointer into the right operand  */
	TypDigit         m2;             /* next digit of the remainder     */
	TypDigit         r1;             /* leading digit of the right oper */
	TypDigit		 r2;             /* next digit of the right operand */
	TypDigit         qi;             /* guessed digit of the quotient   */
	UInt             c;              /* product of two digits           */
	UInt             e;              /* we mult r by 2^e so r1 >= 32768 */
	UInt             rs;             /* size of the right operand       */
	UInt             m01;            /* leading two digits of the rem.  */
	Obj              mod;            /* handle of the remainder bag     */

	/* compute the remainder of two small integers                         */
	if (ARE_INTOBJS(opL, opR))
	{
		/* pathological case first                                         */
		if (opR == INTOBJ_INT(0))
		{
			opR = ErrorReturnObj("Integer operations: <divisor> must be nonzero", 0, 0, "you can replace the integer <divisor> via 'return <divisor>;'");
			return MOD(opL, opR);
		}

		/* get the integer values                                          */
		i = INT_INTOBJ(opL);
		k = INT_INTOBJ(opR);

		/* compute the remainder, make sure we divide only positive numbers*/
		if (0 <= i && 0 <= k)
		{
			i = (i % k); 
		}
		else if (0 <= i && k < 0) 
		{ 
			i = (i % -k); }
		else if (i < 0 && 0 <= k) 
		{ 
			i = (k - (-i % k)) % k; 
		}
		else if (i < 0 && k < 0) 
		{ 
			i = (-k - (-i % -k)) % k; 
		}

		mod = INTOBJ_INT(i);
	}
	/* compute the remainder of a small integer by a large integer         */
	else if (IS_INTOBJ(opL)) 
	{

		/* the small int -(1<<28) mod the large int (1<<28) is 0           */
		if (opL == INTOBJ_INT((UInt)-(Int)(NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			&& TNUM_OBJ(opR) == T_INTPOS && SIZE_INT(opR) == 4
			&& ADDR_INT(opR)[3] == 0
			&& ADDR_INT(opR)[2] == 0
			&& ADDR_INT(opR)[1] == (NR_SMALL_INT_BITS - NR_DIGIT_BITS)
			&& ADDR_INT(opR)[0] == 0)
		{
			mod = INTOBJ_INT(0);
		}
		/* in all other cases the remainder is equal the left operand      */
		else if (0 <= INT_INTOBJ(opL))
		{
			mod = opL;
		}
		else if (TNUM_OBJ(opR) == T_INTPOS)
		{
			mod = SumInt(opL, opR);
		}
		else
		{
			mod = DiffInt(opL, opR);
		}
	}
	/* compute the remainder of a large integer by a small integer         */
	else if (IS_INTOBJ(opR)
			 && INT_INTOBJ(opR) < INTBASE
			 && -(Int)INTBASE <= INT_INTOBJ(opR)) 
	{

		/* pathological case first                                         */
		if (opR == INTOBJ_INT(0)) 
		{
			opR = ErrorReturnObj("Integer operations: <divisor> must be nonzero", 0, 0, "you can replace the integer <divisor> via 'return <divisor>;'");
			return MOD(opL, opR);
		}

		/* get the integer value, make positive                            */
		i = INT_INTOBJ(opR);  if (i < 0)  i = -i;

		/* maybe its trivial                                               */
		if (INTBASE % i == 0) 
		{
			c = ADDR_INT(opL)[0] % i;
		}
		/* otherwise run through the left operand and divide digitwise     */
		else
		{
			l = ADDR_INT(opL) + SIZE_INT(opL) - 1;
			c = 0;
			for (; l >= ADDR_INT(opL); l--)
			{
				c = (c << NR_DIGIT_BITS) + (Int)* l;
				c = c % i;
			}
		}

		/* now c is the result, it has the same sign as the left operand   */
		if (TNUM_OBJ(opL) == T_INTPOS)
		{
			mod = INTOBJ_INT(c);
		}
		else if (c == 0)
		{
			mod = INTOBJ_INT(c);
		}
		else if (0 <= INT_INTOBJ(opR))
		{
			mod = SumInt(INTOBJ_INT(-(Int)c), opR);
		}
		else
		{
			mod = DiffInt(INTOBJ_INT(-(Int)c), opR);
		}

	}
	/* compute the remainder of a large integer modulo a large integer     */
	else
	{
		/* a small divisor larger than one digit isn't handled above       */
		if (IS_INTOBJ(opR))
		{
			if (0 < INT_INTOBJ(opR))
			{
				mod = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
				ADDR_INT(mod)[0] = (TypDigit)(INT_INTOBJ(opR));
				ADDR_INT(mod)[1] = (TypDigit)(INT_INTOBJ(opR) >> NR_DIGIT_BITS);
				opR = mod;
			}
			else 
			{
				mod = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
				ADDR_INT(mod)[0] = (TypDigit)(-INT_INTOBJ(opR));
				ADDR_INT(mod)[1] = (TypDigit)((-INT_INTOBJ(opR)) >> NR_DIGIT_BITS);
				opR = mod;
			}
		}

		/* trivial case first                                              */
		if (SIZE_INT(opL) < SIZE_INT(opR)) 
		{
			if (TNUM_OBJ(opL) == T_INTPOS)
			{
				return opL;
			}
			else if (TNUM_OBJ(opR) == T_INTPOS)
			{
				return SumInt(opL, opR);
			}
			else
			{
				return DiffInt(opL, opR);
			}
		}

		/* copy the left operand into a new bag, this holds the remainder  */
		mod = NewBag(TNUM_OBJ(opL), (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		l = ADDR_INT(opL);
		m = ADDR_INT(mod);
		for (k = SIZE_INT(opL) - 1; k >= 0; k--)
		{
			*m++ = *l++;
		}

		/* get the size of the right operand, and get the leading 2 digits */
		rs = SIZE_INT(opR);
		r = ADDR_INT(opR);
		while (r[rs - 1] == 0) 
		{
			rs--; 
		}

		for (e = 0; ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0) < INTBASE / 2; e++);

		r1 = ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0);
		r2 = ((Int)r[rs - 2] << e) + ((rs >= 3 && e) ? r[rs - 3] >> (NR_DIGIT_BITS - e) : 0);

		/* run through the digits in the quotient                          */
		for (i = SIZE_INT(mod) - SIZE_INT(opR) - 1; i >= 0; i--)
		{

			/* guess the factor                                            */
			m = ADDR_INT(mod) + rs + i;
			m01 = ((INTBASE * m[0] + m[-1]) << e) + (e ? m[-2] >> (NR_DIGIT_BITS - e) : 0);
			if (m01 == 0)
			{ 
				continue;
			}

			m2 = ((Int)m[-2] << e) + ((e && rs + i >= 3) ? m[-3] >> (NR_DIGIT_BITS - e) : 0);

			if (((Int)m[0] << e) + (e ? m[-1] >> (NR_DIGIT_BITS - e) : 0) < r1)
			{
				qi = m01 / r1;
			}
			else 
			{
				qi = INTBASE - 1; 
			}

			while (m01 - (Int)qi * r1 < (Int)INTBASE && INTBASE * (m01 - (Int)qi * r1) + m2 < (Int)qi * r2)
			{
				qi--;
			}

			/* m = m - qi * r;                                             */
			d = 0;
			m = ADDR_INT(mod) + i;
			r = ADDR_INT(opR);

			for (k = 0; k < (Int)rs; ++k, ++m, ++r)
			{
				c = (Int)* m - (Int)qi * *r - (Int)d;
				*m = (TypDigit)c;
				d = -(TypDigit)(c >> NR_DIGIT_BITS);
			}

			c = (Int)* m - d;  *m = (TypDigit)c;  d = -(TypDigit)(c >> NR_DIGIT_BITS);

			/* if we have a borrow then add back                           */
			if (d != 0)
			{
				d = 0;
				m = ADDR_INT(mod) + i;
				r = ADDR_INT(opR);

				for (k = 0; k < (Int)rs; ++k, ++m, ++r)
				{
					c = (Int)* m + (Int)* r + (Int)d;
					*m = (TypDigit)c;
					d = (TypDigit)(c >> NR_DIGIT_BITS);
				}
				c = (Int)* m + d;  *m = (TypDigit)c;  d = (TypDigit)(c >> NR_DIGIT_BITS);
				qi--;
			}

		}

		/* remove the leading zeroes                                       */
		m = ADDR_INT(mod) + SIZE_INT(mod);
		if ((m[-4] == 0 && m[-3] == 0 && m[-2] == 0 && m[-1] == 0) || (SIZE_INT(mod) == 4 && m[-2] == 0 && m[-1] == 0))
		{

			/* find the number of significant digits                       */
			m = ADDR_INT(mod);

			for (k = SIZE_INT(mod); k != 0; k--)
			{
				if (m[k - 1] != 0)
				{
					break;
				}
			}

			/* reduce to small integer if possible, otherwise shrink bag   */

			if (k <= 2 && TNUM_OBJ(mod) == T_INTPOS && (UInt)(INTBASE * m[1] + m[0]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				mod = INTOBJ_INT(INTBASE * m[1] + m[0]);
			}
			else if (k <= 2 && TNUM_OBJ(mod) == T_INTNEG && (UInt)(INTBASE * m[1] + m[0]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				mod = INTOBJ_INT(-(Int)(INTBASE * m[1] + m[0]));
			}
			else
			{
				Resize(mod, (((k + 3) / 4) * 4) * sizeof(TypDigit));
			}
		}

		/* make the representative positive                                  */
		if ((TNUM_OBJ(mod) == T_INT && INT_INTOBJ(mod) < 0) || TNUM_OBJ(mod) == T_INTNEG)
		{
			if (TNUM_OBJ(opR) == T_INTPOS)
			{
				mod = SumInt(mod, opR);
			}
			else
			{
				mod = DiffInt(mod, opR);
			}
		}
	}
	/* return the result                                                   */
	return mod;
}


/****************************************************************************
 **
 *F  QuoInt( <intL>, <intR> )  . . . . . . . . . . . quotient of two integers
 **
 **  'QuoInt' returns the integer part of the two integers <intL> and  <intR>.
 **  'QuoInt' handles operands of type  'T_INT',  'T_INTPOS'  and  'T_INTNEG'.
 **
 **  It can also be used in the cases that both operands  are  small  integers
 **  and the result is a small integer too,  i.e., that  no  overflow  occurs.
 **
 **  Note that this routine is not called from 'EvQuo', the  division  of  two
 **  integers yields  a  rational  and  is  therefor  performed  in  'QuoRat'.
 **  This operation is however available through the internal function 'Quo'.
 */
Obj		QuoInt( Obj opL, Obj opR)
{
	Int              i;              /* loop count, value for small int */
	Int              k;              /* loop count, value for small int */
	TypDigit         d;              /* carry into the next digit       */
	TypDigit		*l;              /* pointer into the left operand   */
	TypDigit		*r;              /* pointer into the right operand  */
	TypDigit		*q;              /* pointer into the quotient       */
	TypDigit         l2;             /* next digit of the left operand  */
	TypDigit         qi;             /* guessed digit of the quotient   */
	TypDigit         r1;             /* leading digit of the right oper */
	TypDigit         r2;             /* next digit of the right operand */
	UInt             c;              /* product of two digits           */
	UInt             e;              /* we mult r by 2^e so r1 >= 32768 */
	UInt             rs;             /* size of the right operand       */
	UInt             l01;            /* leading two digits of the left  */
	Obj              quo;            /* handle of the result bag        */

	/* divide to small integers                                            */
	if (ARE_INTOBJS(opL, opR))
	{
		/* pathological case first                                         */
		if (opR == INTOBJ_INT(0)) 
		{
			opR = ErrorReturnObj("Integer operations: <divisor> must be nonzero", 0, 0, "you can replace the integer <divisor> via 'return <divisor>;'");
			return QUO(opL, opR);
		}

		/* the small int -(1<<28) divided by -1 is the large int (1<<28)   */
		if (opL == INTOBJ_INT(-(Int)(NUM_TO_UINT(1) << NR_SMALL_INT_BITS)) && opR == INTOBJ_INT(-1))
		{
			quo = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
			ADDR_INT(quo)[1] = NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS);
			ADDR_INT(quo)[0] = 0;
			return quo;
		}

		/* get the integer values                                          */
		i = INT_INTOBJ(opL);
		k = INT_INTOBJ(opR);

		/* divide, make sure we divide only positive numbers               */
		if (0 <= i && 0 <= k)
		{
			i = (i / k); 
		}
		else if (0 <= i && k < 0)
		{ 
			i = -(i / -k); 
		}
		else if (i < 0 && 0 <= k)
		{ 
			i = -(-i / k); 
		}
		else if (i < 0 && k < 0)
		{
			i = (-i / -k);
		}
		quo = INTOBJ_INT(i);
	}
	/* divide a small integer by a large one                               */
	else if (IS_INTOBJ(opL))
	{

		/* the small int -(1<<28) divided by the large int (1<<28) is -1   */

		if (opL == INTOBJ_INT(-(Int)(NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			&& TNUM_OBJ(opR) == T_INTPOS && SIZE_INT(opR) == 4
			&& ADDR_INT(opR)[3] == 0
			&& ADDR_INT(opR)[2] == 0
			&& ADDR_INT(opR)[1] == NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS)
			&& ADDR_INT(opR)[0] == 0)
		{
			quo = INTOBJ_INT(-1);
		}
		/* in all other cases the quotient is of course zero               */
		else
		{
			quo = INTOBJ_INT(0);
		}

	}
	/* divide a large integer by a small integer                           */
	else if (IS_INTOBJ(opR)
			 && INT_INTOBJ(opR) < INTBASE
			 && -(Int)INTBASE <= INT_INTOBJ(opR))
	{

		/* pathological case first                                         */
		if (opR == INTOBJ_INT(0))
		{
			opR = ErrorReturnObj("Integer operations: <divisor> must be nonzero", 0, 0, "you can replace the integer <divisor> via 'return <divisor>;'");
			return QUO(opL, opR);
		}

		/* get the integer value, make positive                            */
		i = INT_INTOBJ(opR);  if (i < 0)  i = -i;

		/* allocate a bag for the result and set up the pointers           */
		if ((TNUM_OBJ(opL) == T_INTPOS && 0 < INT_INTOBJ(opR))
			|| (TNUM_OBJ(opL) == T_INTNEG && INT_INTOBJ(opR) < 0))
		{
			quo = NewBag(T_INTPOS, SIZE_OBJ(opL));
		}
		else
		{
			quo = NewBag(T_INTNEG, SIZE_OBJ(opL));
		}

		l = ADDR_INT(opL) + SIZE_INT(opL) - 1;
		q = ADDR_INT(quo) + SIZE_INT(quo) - 1;

		/* run through the left operand and divide digitwise               */
		c = 0;

		for (; l >= ADDR_INT(opL); l--, q--) 
		{
			c = (c << NR_DIGIT_BITS) + (Int)* l;
			*q = (TypDigit)(c / i);
			c = c - i * *q;
			/*N clever compilers may prefer:  c  = c % i;                  */
		}

		/* remove the leading zeroes, note that there can't be more than 5 */
		q = ADDR_INT(quo) + SIZE_INT(quo);
		if (q[-4] == 0 && q[-3] == 0 && q[-2] == 0 && q[-1] == 0) 
		{
			Resize(quo, (SIZE_INT(quo) - 4) * sizeof(TypDigit));
		}

		/* reduce to small integer if possible                             */
		q = ADDR_INT(quo) + SIZE_INT(quo);

		if (SIZE_INT(quo) == 4 && q[-2] == 0 && q[-1] == 0) 
		{
			if (TNUM_OBJ(quo) == T_INTPOS 
				&& (UInt)(INTBASE * q[-3] + q[-4]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				quo = INTOBJ_INT(INTBASE * q[-3] + q[-4]);
			}
			else if (TNUM_OBJ(quo) == T_INTNEG 
				&& ((UInt)(INTBASE * q[-3] + q[-4]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS)))
			{
				quo = INTOBJ_INT(-(Int)(INTBASE * q[-3] + q[-4]));
			}
		}
	}
	/* divide a large integer by a large integer                           */
	else 
	{
		/* a small divisor larger than one digit isn't handled above       */
		if (IS_INTOBJ(opR)) 
		{
			if (0 < INT_INTOBJ(opR))
			{
				quo = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
				ADDR_INT(quo)[0] = (TypDigit)(INT_INTOBJ(opR));
				ADDR_INT(quo)[1] = (TypDigit)(INT_INTOBJ(opR) >> NR_DIGIT_BITS);
				opR = quo;
			}
			else 
			{
				quo = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
				ADDR_INT(quo)[0] = (TypDigit)(-INT_INTOBJ(opR));
				ADDR_INT(quo)[1] = (TypDigit)((-INT_INTOBJ(opR)) >> NR_DIGIT_BITS);
				opR = quo;
			}
		}

		/* trivial case first                                              */
		if (SIZE_INT(opL) < SIZE_INT(opR))
		{
			return INTOBJ_INT(0);
		}

		/* copy the left operand into a new bag, this holds the remainder  */
		quo = NewBag(TNUM_OBJ(opL), (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		l = ADDR_INT(opL);
		q = ADDR_INT(quo);

		for (k = SIZE_INT(opL) - 1; k >= 0; k--)
		{
			*q++ = *l++;
		}

		opL = quo;

		/* get the size of the right operand, and get the leading 2 digits */
		rs = SIZE_INT(opR);
		r = ADDR_INT(opR);

		while (r[rs - 1] == 0) 
		{ 
			rs--; 
		}

		for (e = 0; ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0) < INTBASE / 2; e++);

		r1 = ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0);
		r2 = ((Int)r[rs - 2] << e) + ((e && rs >= 3) ? r[rs - 3] >> (NR_DIGIT_BITS - e) : 0);

		/* allocate a bag for the quotient                                 */
		if (TNUM_OBJ(opL) == TNUM_OBJ(opR))
		{
			quo = NewBag(T_INTPOS, SIZE_OBJ(opL) - SIZE_OBJ(opR));
		}
		else
		{
			quo = NewBag(T_INTNEG, SIZE_OBJ(opL) - SIZE_OBJ(opR));
		}

		/* run through the digits in the quotient                          */
		for (i = SIZE_INT(opL) - SIZE_INT(opR) - 1; i >= 0; i--) 
		{

			/* guess the factor                                            */
			l = ADDR_INT(opL) + rs + i;
			l01 = ((INTBASE * l[0] + l[-1]) << e) + (e ? l[-2] >> (NR_DIGIT_BITS - e) : 0);

			if (l01 == 0) 
			{ 
				continue; 
			}

			l2 = ((Int)l[-2] << e) + ((e && rs + i >= 3) ? l[-3] >> (NR_DIGIT_BITS - e) : 0);

			if (((Int)l[0] << e) + (e ? l[-1] >> (NR_DIGIT_BITS - e) : 0) < r1)
			{
				qi = l01 / r1;
			}
			else
			{
				qi = INTBASE - 1; 
			}

			while (l01 - (Int)qi * r1 < (Int)INTBASE && INTBASE * (l01 - (UInt)qi * r1) + l2 < (UInt)qi * r2)
			{
				qi--;
			}

			/* l = l - qi * r;                                             */
			d = 0;
			l = ADDR_INT(opL) + i;
			r = ADDR_INT(opR);

			for (k = 0; k < (Int)rs; ++k, ++l, ++r) 
			{
				c = *l - (Int)qi * *r - d;  *l = c;
				d = -(TypDigit)(c >> NR_DIGIT_BITS);
			}

			c = (Int)* l - d; d = -(TypDigit)(c >> NR_DIGIT_BITS);

			/* if we have a borrow then add back                           */
			if (d != 0)
			{
				d = 0;
				l = ADDR_INT(opL) + i;
				r = ADDR_INT(opR);

				for (k = 0; k < (Int)rs; ++k, ++l, ++r) 
				{
					c = (Int)* l + (Int)* r + (Int)d;
					*l = (TypDigit)c;
					d = (TypDigit)(c >> NR_DIGIT_BITS);
				}

				c = *l + d; d = (TypDigit)(c >> NR_DIGIT_BITS);
				qi--;
			}
			/* store the digit in the quotient                             */
			ADDR_INT(quo)[i] = qi;
		}
		/* remove the leading zeroes, note that there can't be more than 7 */
		q = ADDR_INT(quo) + SIZE_INT(quo);

		if (SIZE_INT(quo) > 4 && q[-4] == 0 && q[-3] == 0 && q[-2] == 0 && q[-1] == 0) 
		{
			Resize(quo, (SIZE_INT(quo) - 4) * sizeof(TypDigit));
		}

		/* reduce to small integer if possible                             */
		q = ADDR_INT(quo) + SIZE_INT(quo);

		if ((SIZE_INT(quo) == 4) && (q[-2] == 0) && (q[-1] == 0))
		{
			if (TNUM_OBJ(quo) == T_INTPOS && (UInt)(INTBASE * q[-3] + q[-4]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				quo = INTOBJ_INT(INTBASE * q[-3] + q[-4]);
			}
			else if (TNUM_OBJ(quo) == T_INTNEG && (UInt)(INTBASE * q[-3] + q[-4]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				quo = INTOBJ_INT(-(Int)(INTBASE * q[-3] + q[-4]));
			}
		}
	}
	/* return the result                                                   */
	return quo;
}


/****************************************************************************
 **
 *F  RemInt( <intL>, <intR> )  . . . . . . . . . . . remainder of two integers
 **
 **  'RemInt' returns the remainder of the quotient  of  the  integers  <intL>
 **  and <intR>.  'RemInt' handles operands of type  'T_INT',  'T_INTPOS'  and
 **  'T_INTNEG'.
 **
 **  Note that the remainder is different from the value returned by the 'mod'
 **  operator which is always positive.
 **
 **  'RemInt' is called from 'FunRemInt'.
 */
Obj		RemInt(Obj opL, Obj opR)
{
	Int          i;              /* loop count, value for small int */
	Int          k;              /* loop count, value for small int */
	TypDigit	 d;              /* carry into the next digit       */
	TypDigit	*l;              /* pointer into the left operand   */
	TypDigit	*r;              /* pointer into the right operand  */
	TypDigit	*m;              /* pointer into the remainder      */
	TypDigit     m2;             /* next digit of the remainder     */
	TypDigit     r1;             /* leading digit of the right oper */
	TypDigit     r2;             /* next digit of the right operand */
	TypDigit     qi;             /* guessed digit of the quotient   */
	UInt         c;              /* product of two digits           */
	UInt         e;              /* we mult r by 2^e so r1 >= 32768 */
	UInt         rs;             /* size of the right operand       */
	UInt         m01;            /* leading two digits of the rem.  */
	Obj          rem;            /* handle of the remainder bag     */

	/* compute the remainder of two small integers                         */
	if (ARE_INTOBJS(opL, opR))
	{
		/* pathological case first                                         */
		if (opR == INTOBJ_INT(0))
		{
			opR = ErrorReturnObj("Integer operations: <divisor> must be nonzero", 0, 0, "you can replace the integer <divisor> via 'return <divisor>;'");
			return QUO(opL, opR);
		}

		/* get the integer values                                          */
		i = INT_INTOBJ(opL);
		k = INT_INTOBJ(opR);

		/* compute the remainder, make sure we divide only positive numbers*/
		if (0 <= i && 0 <= k) 
		{ 
			i = (i % k);
		}
		else if (0 <= i && k < 0)
		{
			i = (i % -k); 
		}
		else if (i < 0 && 0 <= k)
		{ 
			i = -(-i % k); 
		}
		else if (i < 0 && k < 0)
		{ 
			i = -(-i % -k); 
		}

		rem = INTOBJ_INT(i);
	}
	/* compute the remainder of a small integer by a large integer         */
	else if (IS_INTOBJ(opL))
	{
		/* the small int -(1<<28) rem the large int (1<<28) is 0           */
		if (opL == INTOBJ_INT(-(Int)(NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			&& TNUM_OBJ(opR) == T_INTPOS && SIZE_INT(opR) == 4
			&& ADDR_INT(opR)[3] == 0
			&& ADDR_INT(opR)[2] == 0
			&& ADDR_INT(opR)[1] == NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS)
			&& ADDR_INT(opR)[0] == 0)
		{
			rem = INTOBJ_INT(0);
		}
		/* in all other cases the remainder is equal the left operand      */
		else
		{
			rem = opL;
		}
	}
	/* compute the remainder of a large integer by a small integer         */
	else if (IS_INTOBJ(opR) && INT_INTOBJ(opR) < INTBASE && -(Int)INTBASE <= INT_INTOBJ(opR))
	{
		/* pathological case first                                         */
		if (opR == INTOBJ_INT(0))
		{
			opR = ErrorReturnObj("Integer operations: <divisor> must be nonzero", 0, 0,	"you can replace the integer <divisor> via 'return <divisor>;'");
			return QUO(opL, opR);
		}

		/* get the integer value, make positive                            */
		i = INT_INTOBJ(opR);  if (i < 0)  i = -i;

		/* maybe its trivial                                               */
		if (INTBASE % i == 0) 
		{
			c = ADDR_INT(opL)[0] % i;
		}
		/* otherwise run through the left operand and divide digitwise     */
		else 
		{
			l = ADDR_INT(opL) + SIZE_INT(opL) - 1;
			c = 0;

			for (; l >= ADDR_INT(opL); l--) 
			{
				c = (c << NR_DIGIT_BITS) + (Int)* l;
				c = c % i;
			}
		}

		/* now c is the result, it has the same sign as the left operand   */
		if (TNUM_OBJ(opL) == T_INTPOS)
		{
			rem = INTOBJ_INT(c);
		}
		else
		{
			rem = INTOBJ_INT(-(Int)c);
		}
	}
	/* compute the remainder of a large integer modulo a large integer     */
	else 
	{

		/* a small divisor larger than one digit isn't handled above       */
		if (IS_INTOBJ(opR))
		{
			if (0 < INT_INTOBJ(opR))
			{
				rem = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
				ADDR_INT(rem)[0] = (TypDigit)(INT_INTOBJ(opR));
				ADDR_INT(rem)[1] = (TypDigit)(INT_INTOBJ(opR) >> NR_DIGIT_BITS);
				opR = rem;
			}
			else 
			{
				rem = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
				ADDR_INT(rem)[0] = (TypDigit)(-INT_INTOBJ(opR));
				ADDR_INT(rem)[1] = (TypDigit)((-INT_INTOBJ(opR)) >> NR_DIGIT_BITS);
				opR = rem;
			}
		}

		/* trivial case first                                              */
		if (SIZE_INT(opL) < SIZE_INT(opR))
		{
			return opL;
		}

		/* copy the left operand into a new bag, this holds the remainder  */
		rem = NewBag(TNUM_OBJ(opL), (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		l = ADDR_INT(opL);
		m = ADDR_INT(rem);
		for (k = SIZE_INT(opL) - 1; k >= 0; k--)
		{
			*m++ = *l++;
		}

		/* get the size of the right operand, and get the leading 2 digits */
		rs = SIZE_INT(opR);
		r = ADDR_INT(opR);

		while (r[rs - 1] == 0) 
		{ 
			rs--; 
		}

		for (e = 0; ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0)< INTBASE / 2; e++);

		r1 = ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0);
		r2 = ((Int)r[rs - 2] << e) + ((e && rs >= 3) ? r[rs - 3] >> (NR_DIGIT_BITS - e) : 0);

		/* run through the digits in the quotient                          */
		for (i = SIZE_INT(rem) - SIZE_INT(opR) - 1; i >= 0; i--)
		{
			/* guess the factor                                            */
			m = ADDR_INT(rem) + rs + i;
			m01 = ((INTBASE * m[0] + m[-1]) << e) + (e ? m[-2] >> (NR_DIGIT_BITS - e) : 0);

			if (m01 == 0)
			{ 
				continue;
			}

			m2 = ((Int)m[-2] << e) + ((e && rs + i >= 3) ? m[-3] >> (NR_DIGIT_BITS - e) : 0);

			if (((Int)m[0] << e) + (e ? m[-1] >> (NR_DIGIT_BITS - e) : 0) < r1)
			{
				qi = m01 / r1;
			}
			else 
			{ 
				qi = INTBASE - 1; 
			}
			while (m01 - (Int)qi * r1 < (Int)INTBASE && INTBASE * (m01 - (Int)qi * r1) + m2 < (Int)qi * r2)
			{
				qi--;
			}

			/* m = m - qi * r;                                             */
			d = 0;
			m = ADDR_INT(rem) + i;
			r = ADDR_INT(opR);

			for (k = 0; k < (Int)rs; ++k, ++m, ++r) 
			{
				c = (Int)* m - (Int)qi * *r - (Int)d;
				*m = (TypDigit)c;
				d = -(TypDigit)(c >> NR_DIGIT_BITS);
			}

			c = (Int)* m - d;  *m = (TypDigit)c;  d = -(TypDigit)(c >> NR_DIGIT_BITS);

			/* if we have a borrow then add back                           */
			if (d != 0)
			{
				d = 0;
				m = ADDR_INT(rem) + i;
				r = ADDR_INT(opR);

				for (k = 0; k < (Int)rs; ++k, ++m, ++r)
				{
					c = (Int)* m + (Int)* r + (Int)d;
					*m = (TypDigit)c;
					d = (TypDigit)(c >> NR_DIGIT_BITS);
				}

				c = (Int)* m + d;  
				*m = (TypDigit)c; 
				d = (TypDigit)(c >> NR_DIGIT_BITS);
				qi--;
			}

		}

		/* remove the leading zeroes                                       */
		m = ADDR_INT(rem) + SIZE_INT(rem);

		if ((m[-4] == 0 && m[-3] == 0 && m[-2] == 0 && m[-1] == 0) || (SIZE_INT(rem) == 4 && m[-2] == 0 && m[-1] == 0)) 
		{
			/* find the number of significant digits                       */
			m = ADDR_INT(rem);

			for (k = SIZE_INT(rem); k != 0; k--)
			{
				if (m[k - 1] != 0)
				{
					break;
				}
			}

			/* reduce to small integer if possible, otherwise shrink bag   */
			if (k <= 2 && TNUM_OBJ(rem) == T_INTPOS && (UInt)(INTBASE * m[1] + m[0]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				rem = INTOBJ_INT(INTBASE * m[1] + m[0]);
			}
			else if (k <= 2 && TNUM_OBJ(rem) == T_INTNEG && (UInt)(INTBASE * m[1] + m[0]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				rem = INTOBJ_INT(-(Int)(INTBASE * m[1] + m[0]));
			}
			else
			{
				Resize(rem, (((k + 3) / 4) * 4) * sizeof(TypDigit));
			}
		}

	}
	/* return the result                                                   */
	return rem;
}


/****************************************************************************
 **
 *F  GcdInt( <opL>, <opR> )  . . . . . . . . . . . . . . . gcd of two integers
 **
 **  'GcdInt' returns the gcd of the two integers <opL> and <opR>.
 **
 **  It is called from 'FunGcdInt' and the rational package.
 */
Obj		GcdInt(Obj opL,Obj opR)
{
	Int          i;              /* loop count, value for small int */
	Int          k;              /* loop count, value for small int */
	TypDigit     d;              /* carry into the next digit       */
	TypDigit	*l;              /* pointer into the left operand   */
	TypDigit	*r;              /* pointer into the right operand  */
	TypDigit     l2;             /* next digit of the remainder     */
	TypDigit     r1;             /* leading digit of the right oper */
	TypDigit     r2;             /* next digit of the right operand */
	TypDigit     qi;             /* guessed digit of the quotient   */
	UInt         c;              /* product of two digits           */
	UInt         e;              /* we mult r by 2^e so r1 >= 32768 */
	UInt         ls;             /* size of the left operand        */
	UInt         rs;             /* size of the right operand       */
	UInt         l01;            /* leading two digits of the rem.  */
	Obj          gcd;            /* handle of the result            */

	/* compute the gcd of two small integers                               */
	if (ARE_INTOBJS(opL, opR))
	{
		/* get the integer values, make them positive                      */
		i = INT_INTOBJ(opL);  if (i < 0)  i = -i;
		k = INT_INTOBJ(opR);  if (k < 0)  k = -k;

		/* compute the gcd using Euclids algorithm                         */
		while (k != 0)
		{
			c = k;
			k = i % k;
			i = c;
		}

		/* now i is the result                                             */
		if (i == (NUM_TO_UINT(1) << NR_SMALL_INT_BITS)) 
		{
			gcd = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
			ADDR_INT(gcd)[0] = (TypDigit)i;
			ADDR_INT(gcd)[1] = (TypDigit)(i >> NR_DIGIT_BITS);
		}
		else 
		{
			gcd = INTOBJ_INT(i);
		}

	}
	/* compute the gcd of a small and a large integer                      */
	else if ((IS_INTOBJ(opL) && INT_INTOBJ(opL) < INTBASE && -(Int)INTBASE <= INT_INTOBJ(opL)) 
		|| (IS_INTOBJ(opR) && INT_INTOBJ(opR) < INTBASE && -(Int)INTBASE <= INT_INTOBJ(opR))) 
	{

		/* make the right operand the small one                            */
		if (IS_INTOBJ(opL))
		{
			gcd = opL;  opL = opR;  opR = gcd;
		}

		/* maybe it's trivial                                              */
		if (opR == INTOBJ_INT(0)) 
		{
			if (TNUM_OBJ(opL) == T_INTNEG) 
			{
				/* If opL is negative, change the sign.  We do this by    */
				/* copying opL into a bag of type T_INTPOS.  Note that    */
				/* opL is a large negative number, so it cannot be the    */
				/* the negative of 1 << NR_SMALL_INT_BITS.                */
				gcd = NewBag(T_INTPOS, SIZE_OBJ(opL));
				l = ADDR_INT(opL); r = ADDR_INT(gcd);
				for (k = SIZE_INT(opL); k != 0; k--)* r++ = *l++;

				return gcd;
			}
			else return opL;
		}

		/* get the right operand value, make it positive                   */
		i = INT_INTOBJ(opR);  if (i < 0)  i = -i;

		/* do one remainder operation                                      */
		l = ADDR_INT(opL) + SIZE_INT(opL) - 1;
		c = 0;

		for (; l >= ADDR_INT(opL); l--) 
		{
			c = (c << NR_DIGIT_BITS) + (Int)* l;
			c = c % i;
		}

		k = c;

		/* compute the gcd using Euclids algorithm                         */
		while (k != 0) 
		{
			c = k;
			k = i % k;
			i = c;
		}

		/* now i is the result                                             */
		if (i == (NUM_TO_UINT(1) << NR_SMALL_INT_BITS)) 
		{
			gcd = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
			ADDR_INT(gcd)[0] = 0;
			ADDR_INT(gcd)[1] = NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS);
		}
		else 
		{
			gcd = INTOBJ_INT(i);
		}
	}
	/* compute the gcd of two large integers                               */
	else 
	{
		/* a small divisor larger than one digit isn't handled above       */
		if (IS_INTOBJ(opL))
		{
			if (0 < INT_INTOBJ(opL))
			{
				gcd = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
				ADDR_INT(gcd)[0] = (TypDigit)(INT_INTOBJ(opL));
				ADDR_INT(gcd)[1] = (TypDigit)(INT_INTOBJ(opL) >> NR_DIGIT_BITS);
				opL = gcd;
			}
			else 
			{
				gcd = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
				ADDR_INT(gcd)[0] = (TypDigit)(-INT_INTOBJ(opL));
				ADDR_INT(gcd)[1] = (TypDigit)((-INT_INTOBJ(opL)) >> NR_DIGIT_BITS);
				opL = gcd;
			}
		}

		/* a small dividend larger than one digit isn't handled above       */
		if (IS_INTOBJ(opR))
		{
			if (0 < INT_INTOBJ(opR)) 
			{
				gcd = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
				ADDR_INT(gcd)[0] = (TypDigit)(INT_INTOBJ(opR));
				ADDR_INT(gcd)[1] = (TypDigit)(INT_INTOBJ(opR) >> NR_DIGIT_BITS);
				opR = gcd;
			}
			else 
			{
				gcd = NewBag(T_INTNEG, 4 * sizeof(TypDigit));
				ADDR_INT(gcd)[0] = (TypDigit)(-INT_INTOBJ(opR));
				ADDR_INT(gcd)[1] = (TypDigit)((-INT_INTOBJ(opR)) >> NR_DIGIT_BITS);
				opR = gcd;
			}
		}

		/* copy the left operand into a new bag                            */
		gcd = NewBag(T_INTPOS, (SIZE_INT(opL) + 4) * sizeof(TypDigit));
		l = ADDR_INT(opL);
		r = ADDR_INT(gcd);

		for (k = SIZE_INT(opL) - 1; k >= 0; k--)
		{
			*r++ = *l++;
		}

		opL = gcd;

		/* get the size of the left operand                                */
		ls = SIZE_INT(opL);
		l = ADDR_INT(opL);
		while (ls >= 1 && l[ls - 1] == 0)
		{
			ls--;
		}

		/* copy the right operand into a new bag                           */
		gcd = NewBag(T_INTPOS, (SIZE_INT(opR) + 4) * sizeof(TypDigit));
		r = ADDR_INT(opR);
		l = ADDR_INT(gcd);

		for (k = SIZE_INT(opR) - 1; k >= 0; k--)
		{
			*l++ = *r++;
		}

		opR = gcd;

		/* get the size of the right operand                               */
		rs = SIZE_INT(opR);
		r = ADDR_INT(opR);

		while (rs >= 1 && r[rs - 1] == 0)
		{
			rs--; 
		}

		/* repeat while the right operand is large                         */
		while (rs >= 2) 
		{

			/* get the leading two digits                                  */
			for (e = 0;
				 ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0)< INTBASE / 2;
				 e++);

			r1 = ((Int)r[rs - 1] << e) + (e ? r[rs - 2] >> (NR_DIGIT_BITS - e) : 0);
			r2 = ((Int)r[rs - 2] << e) + ((e && rs >= 3) ? r[rs - 3] >> (NR_DIGIT_BITS - e) : 0);

			/* run through the digits in the quotient                      */
			for (i = ls - rs; i >= 0; i--) 
			{

				/* guess the factor                                        */
				l = ADDR_INT(opL) + rs + i;
				l01 = ((INTBASE * l[0] + l[-1]) << e) + (e ? l[-2] >> (NR_DIGIT_BITS - e) : 0);

				if (l01 == 0)
				{
					continue; 
				}

				l2 = ((Int)l[-2] << e) + ((e && rs + i >= 3) ? l[-3] >> (NR_DIGIT_BITS - e) : 0);

				if (((Int)l[0] << e) + (e ? l[-1] >> (NR_DIGIT_BITS - e) : 0) < r1)
				{
					qi = l01 / r1;
				}
				else
				{
					qi = INTBASE - 1; 
				}

				while (l01 - (Int)qi * r1 < INTBASE && (INTBASE * (l01 - (UInt)qi * r1) + l2) < ((UInt)qi * r2))
				{
					qi--;
				}

				/* l = l - qi * r;                                         */
				d = 0;
				l = ADDR_INT(opL) + i;
				r = ADDR_INT(opR);

				for (k = 0; k < (Int)rs; ++k, ++l, ++r) 
				{
					c = (Int)* l - (Int)qi * *r - (Int)d;
					*l = (TypDigit)c;
					d = -(TypDigit)(c >> NR_DIGIT_BITS);
				}

				c = *l - d; 
				*l = (TypDigit)c; 
				d = -(TypDigit)(c >> NR_DIGIT_BITS);

				/* if we have a borrow then add back                       */
				if (d != 0)
				{
					d = 0;
					l = ADDR_INT(opL) + i;
					r = ADDR_INT(opR);
					for (k = 0; k < (Int)rs; ++k, ++l, ++r) 
					{
						c = (Int)* l + (Int)* r + (Int)d;
						*l = (TypDigit)c;
						d = (TypDigit)(c >> NR_DIGIT_BITS);
					}

					c = *l + d; 
					*l = (TypDigit)c;  
					d = (TypDigit)(c >> NR_DIGIT_BITS);
					qi--;
				}
			}

			/* exchange the two operands                                   */
			gcd = opL;  opL = opR;  opR = gcd;
			ls = rs;

			/* get the size of the right operand                           */
			rs = SIZE_INT(opR);
			r = ADDR_INT(opR);
			while (rs >= 1 && r[rs - 1] == 0) 
			{
				rs--;
			}

		}

		/* if the right operand is zero now, the left is the gcd           */
		if (rs == 0) 
		{

			/* remove the leading zeroes                                   */
			l = ADDR_INT(opL) + SIZE_INT(opL);

			if ((l[-4] == 0 && l[-3] == 0 && l[-2] == 0 && l[-1] == 0) || (SIZE_INT(opL) == 4 && l[-2] == 0 && l[-1] == 0)) {

				/* find the number of significant digits                   */
				l = ADDR_INT(opL);

				for (k = SIZE_INT(opL); k != 0; k--) 
				{
					if (l[k - 1] != 0)
					{
						break;
					}
				}

				/* reduce to small integer if possible, otherwise shrink b */
				if ((k <= 2) && (TNUM_OBJ(opL) == T_INTPOS) && ((UInt)(INTBASE * l[1] + l[0]) < (NUM_TO_UINT(1) << NR_SMALL_INT_BITS)))
				{
					opL = INTOBJ_INT(INTBASE * l[1] + l[0]);
				}
				else if ((k <= 2) && (TNUM_OBJ(opL) == T_INTNEG) && ((UInt)(INTBASE * l[1] + l[0]) <= (NUM_TO_UINT(1) << NR_SMALL_INT_BITS)))
				{
					opL = INTOBJ_INT(-(Int)(INTBASE * l[1] + l[0]));
				}
				else
				{
					Resize(opL, (((k + 3) / 4) * 4) * sizeof(TypDigit));
				}
			}
			gcd = opL;
		}
		/* otherwise handle one large and one small integer as above       */
		else
		{

			/* get the right operand value, make it positive               */
			i = r[0];

			/* do one remainder operation                                  */
			l = ADDR_INT(opL) + SIZE_INT(opL) - 1;
			c = 0;

			for (; l >= ADDR_INT(opL); l--) 
			{
				c = (c << NR_DIGIT_BITS) + (Int)* l;
				c = c % i;
			}

			k = c;

			/* compute the gcd using Euclids algorithm                     */
			while (k != 0) 
			{
				c = k;
				k = i % k;
				i = c;
			}

			/* now i is the result                                         */
			if (i == (NUM_TO_UINT(1) << NR_SMALL_INT_BITS))
			{
				gcd = NewBag(T_INTPOS, 4 * sizeof(TypDigit));
				ADDR_INT(gcd)[0] = 0;
				ADDR_INT(gcd)[1] = NUM_TO_UINT(1) << (NR_SMALL_INT_BITS - NR_DIGIT_BITS);
			}
			else 
			{
				gcd = INTOBJ_INT(i);
			}
		}
	}
	/* return the result                                                   */
	return gcd;
}


/****************************************************************************
 **
 *F  HashBag(<self>,<obj>,<factor>,<offset>,<maxlen>)
 **
 **  Takes an non-immediate object and a small  integer <int>  and  computes a
 **  hash value for the contents of the bag.
 **
 **  For this to be usable in algorithms, objects of this kind must  be stored
 **  uniquely internally.
 **
 **  The offset and the maximum number of bytes to process both count in bytes.
 **  The values passed to these parameters might depend on the word length of
 **  the computer.  A <maxlen> value of -1 indicates infinity.
 */

Obj		HashBag( Obj opL, Obj opR, Obj opO, Obj opM)
{
	Char	*ptr;
	Int		 i;
	Int		 n;
	Int		 m;
	Int		 sum;
	Int		 offs;
	Int		 modulus;

	modulus = 1 << 28; /* might want to change for 64 bit machines? */

	/* check the arguments                                                 */
	while (TNUM_OBJ(opR) != T_INT) 
	{
		opR = ErrorReturnObj("HASHKEY_BAG: <factor> must be a small integer (not a %s)", (Int)TNAM_OBJ(opR), 0, "you can replace <factor> via 'return <factor>;'");
	}

	while (TNUM_OBJ(opO) != T_INT)
	{
		opR = ErrorReturnObj("HASHKEY_BAG: <offset> must be a small integer (not a %s)", (Int)TNAM_OBJ(opO), 0, "you can replace <offset> via 'return <offset>;'");
	}

	while (TNUM_OBJ(opM) != T_INT)
	{
		opR = ErrorReturnObj("HASHKEY_BAG: <maxlen> must be a small integer (not a %s)", (Int)TNAM_OBJ(opM), 0, "you can replace <maxlen> via 'return <maxlen>;'");
	}

	sum = 0;
	/* start byte plus offset */
	offs = INT_INTOBJ(opO);
	ptr = (Char*)PTR_BAG(opL) + offs;
	n = SIZE_OBJ(opL) - offs;

	/* maximal number of bytes to read */
	offs = INT_INTOBJ(opM);
	if ((n > offs) && (offs != -1)) 
	{
		n = offs;
	};

	m = INT_INTOBJ(opR);

	for (i = 0; i < n; i++)
	{
		sum = ((sum * m) + (Int)(*ptr++));
		/* Pr("%d %d\n",(Int)*ptr,sum); */
	}
	sum = sum % modulus;

	return INTOBJ_INT(sum);
}

/****************************************************************************
 **
 *F  SaveInt( <int> )
 **
 **  Since the type is saved, we don't need to worry about sign
 */
/*
  void SaveInt( Obj bigint) {
  TypDigit *ptr;
  UInt i;
  ptr = (TypDigit *)PTR_BAG(bigint);
  for (i = 0; i < SIZE_INT(bigint); i++)
  #ifdef SYS_IS_64_BIT
  SaveUInt4(*ptr++);
  #else
  SaveUInt2(*ptr++);
  #endif
  return;
  }
*/

/****************************************************************************
 **
 *F  LoadInt( <int> )
 **
 **  Since the type is loaded, we don't need to worry about sign
 */
/*
  void LoadInt( Obj bigint) {
  TypDigit *ptr;
  UInt i;
  ptr = (TypDigit *)PTR_BAG(bigint);
  for (i = 0; i < SIZE_INT(bigint); i++)
  #ifdef SYS_IS_64_BIT
  *ptr++ = LoadUInt4();
  #else
  *ptr++ = LoadUInt2();
  #endif
  return;
  }
*/

/****************************************************************************
 **
 *F * * * * * * * * * * * * * initialize package * * * * * * * * * * * * * * *
 */

/****************************************************************************
 **
 *F  InitIntImplementation( ) . . . . . . . initialise library data structures
 */
void            InitIntImplementation(void)
{
	UInt                t1, t2;

	/* install the marking functions                                       */
	InfoBags[T_INT].name = "integer";
#ifdef SYS_IS_64_BIT
	InfoBags[T_INTPOS].name = "integer (>= 2^60)";
	InfoBags[T_INTNEG].name = "integer (< -2^60)";
#else
	InfoBags[T_INTPOS].name = "integer (>= 2^28)";
	InfoBags[T_INTNEG].name = "integer (< -2^28)";
#endif
	/*  gasman 4
		InitMarkFuncBags( T_INTPOS, MarkNoSubBags );
		InitMarkFuncBags( T_INTNEG, MarkNoSubBags );
	*/
	/* install the binary arithmetic methods                               */
	for (t1 = T_INT; t1 <= T_INTNEG; t1++)
	{
		for (t2 = T_INT; t2 <= T_INTNEG; t2++)
		{
			TabSum[t1][t2] = SumInt;
			TabDiff[t1][t2] = DiffInt;
			TabProd[t1][t2] = ProdInt;
			TabMod[t1][t2] = ModInt;
			TabPow[t1][t2] = PowInt;
			TabEq[t1][t2] = EqInt;
			TabLt[t1][t2] = LtInt;
		}
	}

	InstPrFunc(T_INT, PrInteger);
	InstPrFunc(T_INTPOS, PrInteger);
	InstPrFunc(T_INTNEG, PrInteger);

	InstEvFunc(T_INT, EvInt);
	InstEvFunc(T_INTPOS, EvInt);
	InstEvFunc(T_INTNEG, EvInt);

}


/****************************************************************************
 **
 *E  integer.c . . . . . . . . . . . . . . . . . . . . . . . . . . . ends here
 */
