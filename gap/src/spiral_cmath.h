/* C Math Library Interface */

/****************************************************************************
**
*F  IsIntPair(<hdList>) . . . . . . test whether an object is an integer pair
**
**  'IsIntPair' returns 1 if the list <hdList> is a list with 2 elements, 
**  and each of these 2 is an integer.
*/
Int            IsIntPair ( Bag hdList );

/****************************************************************************
**
*F  DoubleToIntPair(<hdResult>, <double>)  . . . .  convert double to IntPair
**
**  'DoubleToIntPair' converts  the  double precision  floating-point  number
**  to a   mantissa,  exponent pair,  stored  in  IntPair  handle. Result  is
**  constructed by directly modifying hdResult,  which is  assumed  to be  an
**  IntPair.
**  
*/
void            DoubleToIntPair ( Bag hdResult, double d );

/****************************************************************************
**
*F  IntPairToDouble(<hdList>) . . . . .  . . . . .  convert IntPair to double
**
**  'IntPairToDouble' converts the (mantissa, exponent) pair  in hdList to  a 
**  double precision floating point number.
**
**  No error checking is done, so hdList must be guaranteed to be a list of 2
**  integers of the correct size.
*/
double          IntPairToDouble ( Bag hdList );

/****************************************************************************
**
*F  InitSPIRAL_CMath() . . . . . . . . . initializes C Math library interface
**
**  'InitSPIRA_CMath' initializes C Math library interface used by SPIRAL
*/
void            InitSPIRAL_CMath ( void );
