/****************************************************************************
**
*A  vector.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions  that mainly  operate  on vectors  whose
**  elements are integers, rationals, or elements from cyclotomic fields.  As
**  vectors are special lists many things are done in the list package.
**
**  A *vector* is a list that has no holes,  and whose elements all come from
**  a common field.  For the full definition of vectors see chapter "Vectors"
**  in  the {\GAP} manual.   Read also about "More   about Vectors" about the
**  vector flag and the compact representation of vectors over finite fields.
**
**  A list  that  is  known  to be a vector is represented  by  a bag of type
**  'T_VECTOR', which  has  exactely the same representation  as bags of type
**  'T_LIST'.  As a matter of fact the functions in this  file do not  really
**  know   how   this   representation   looks,   they   use    the    macros
**  'SIZE_PLEN_PLIST',   'PLEN_SIZE_PLIST',   'LEN_PLIST',   'SET_LEN_PLIST',
**  'ELM_PLIST', and 'SET_ELM_PLIST' exported by the plain list package.
**
**  Note  that  a list  represented by  a  bag  of type 'T_LIST',  'T_SET' or
**  'T_RANGE' might still be a vector over the rationals  or cyclotomics.  It
**  is just that the kernel does not known this.
**
**  This  package only consists  of  the  functions 'LenVector', 'ElmVector',
**  'ElmsVector',   AssVector',   'AsssVector',  'PosVector',  'PlainVector',
**  'IsDenseVector',  'IsPossVector', 'EqVector', and  'LtVector'.  They  are
**  the  functions  required by  the  generic  lists  package.   Using  these
**  functions the  other  parts of  the  {\GAP} kernel can access and  modify
**  vectors without actually being aware that they are dealing with a vector.
**
*/


/****************************************************************************
**
*F  LenVector(<hdList>) . . . . . . . . . . . . . . . . .  length of a vector
**
**  'LenVector' returns the length of the vector <hdList> as a C integer.
**
**  'LenVector' is the function in 'TabLenList' for vectors.
*/
extern  Int            LenVector(Bag hdList);


/****************************************************************************
**
*F  ElmVector(<hdList>,<pos>) . . . . . . . . . select an element of a vector
**
**  'ElmVector' selects the element at position <pos> of the vector <hdList>.
**  It is the responsibility of the caller to ensure that <pos> is a positive
**  integer.  An  error is signalled if <pos>  is  larger than the  length of
**  <hdList>.
**
**  'ElmfVector' does  the same thing than 'ElmList', but need not check that
**  <pos>  is less than  or  equal to the  length of  <hdList>, this  is  the
**  responsibility of the caller.
**
**  'ElmVector' is the function in 'TabElmList' for vectors.  'ElmfVector' is
**  the  function  in  'TabElmfList', 'TabElmlList',  and  'TabElmrList'  for
**  vectors.
*/
extern  Bag       ElmVector (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmfVector (
            Bag           hdList,
            Int                pos );


/****************************************************************************
**
*F  ElmsVector(<hdList>,<hdPoss>) . . . . . .  select a sublist from a vector
**
**  'ElmsVector' returns a new list containing the elements  at the  position
**  given  in  the  list  <hdPoss>  from  the  vector  <hdList>.  It  is  the
**  responsibility  of  the  caller  to  ensure that  <hdPoss> is  dense  and
**  contains only positive integers.   An error is signalled if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsVector' is the function in 'TabElmsList' for vectors.
*/
extern  Bag       ElmsVector (
            Bag           hdList,
            Bag           hdPoss );


/****************************************************************************
**
*F  AssVector(<hdList>,<pos>,<hdVal>) . . . . . . . . . .  assign to a vector
**
**  'AssVector' assigns the  value  <hdVal>  to the  vector  <hdList>  at the
**  position <pos>.   It is  the responsibility of the  caller to ensure that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  If the position is larger  then the length of the vector <list>, the list
**  is automatically  extended.  To avoid  making this too often, the  bag of
**  the list is extended by at least '<length>/8 +  4' handles.  Thus in  the
**  loop
**
**      l := [];  for i in [1..1024]  do l[i] := i^2;  od;
**
**  the list 'l' is extended only 32 times not 1024 times.
**
**  'AssVector' is the function in 'TabAssList' for vectors.
**
**  'AssVector' simply converts  the  vector into a plain list, and then does
**  the same  stuff as  'AssPlist'.   This is because  a  vector is  not very
**  likely to stay a vector after the assignment.
*/
extern  Bag       AssVector (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


/****************************************************************************
**
*F  AsssVector(<hdList>,<hdPoss>,<hdVals>)assign several elements to a vector
**
**  'AsssVector' assignes the values from the  list <hdVals> at the positions
**  given  in  the  list  <hdPoss>  to   the  vector  <hdList>.   It  is  the
**  responsibility  of the  caller to  ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssVector' is the function in 'TabAsssList' for vectors.
**
**  'AsssVector' simply converts the vector to a plain list and then does the
**  same stuff as 'AsssPlist'.  This is because a vector  is not  very likely
**  to stay a vector after the assignment.
*/
extern  Bag       AsssVector (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


/****************************************************************************
**
*F  PosVector(<hdList>,<hdVal>,<start>) .  position of an element in a vector
**
**  'PosVector' returns  the  position of the  value  <hdVal>  in the  vector
**  <hdList> after the first position <start> as  a C integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosVector' is the function in 'TabPosList' for vectors.
*/
extern  Int            PosVector (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


/****************************************************************************
**
*F  PlainVector(<hdList>) . . . . . . . . .  convert a vector to a plain list
**
**  'PlainVector'  converts the vector  <hdList> to  a plain list.   Not much
**  work.
**
**  'PlainVector' is the function in 'TabPlainList' for vectors.
*/
extern  void            PlainVector (
            Bag           hdList );


/****************************************************************************
**
*F  IsDenseVector(<hdList>) . . . . . .  dense list test function for vectors
**
**  'IsDenseVector' returns 1, since every vector is dense.
**
**  'IsDenseVector' is the function in 'TabIsDenseList' for vectors.
*/
extern  Int            IsDenseVector (
            Bag           hdList );


/****************************************************************************
**
*F  IsPossVector(<hdList>)  . . . .  positions list test function for vectors
**
**  'IsPossVector'  returns  1  if  the  vector  <hdList>  is  a  dense  list
**  containing only positive integers, and 0 otherwise.
**
**  'IsPossVector' is the function in 'TabIsPossList' for vectors.
*/
extern  Int            IsPossVector (
            Bag           hdList );


/****************************************************************************
**
*F  IsXTypeVector(<hdList>) . . . . . . . . . . .  test if a list is a vector
**
**  'IsXTypeVector' returns  1  if  the  list  <hdList>  is a  vector  and  0
**  otherwise.   As  a  sideeffect  the  type  of  the  list  is  changed  to
**  'T_VECTOR'.
**
**  'IsXTypeVector' is the function in 'TabIsXTypeList' for vectors.
*/
extern  Int            IsXTypeVector (
            Bag           hdList );


/****************************************************************************
**
*F  IsXTypeMatrix(<hdList>) . . . . . . . . . . .  test if a list is a matrix
**
**  'IsXTypeMatrix'  returns  1  if  the  list <hdList>  is  a  matrix and  0
**  otherwise.   As  a  sideeffect  the  type  of  the  rows  is  changed  to
**  'T_VECTOR'.
**
**  'IsXTypeMatrix' is the function in 'TabIsXTypeList' for matrices.
*/
extern  Int            IsXTypeMatrix (
            Bag           hdList );


/****************************************************************************
**
*F  EqVector(<hdL>,<hdR>) . . . . . . . . . . . test if two vectors are equal
**
**  'EqVector'  returns 'true' if  the two vectors <hdL> and  <hdR> are equal
**  and 'false' otherwise.
**
**  Is called from the 'EQ' binop so both operands are already evaluated.
*/
extern  Bag       EqVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  LtVector(<hdL>,<hdR>) . . . . . . . . . . . test if two vectors are equal
**
**  'LtList' returns 'true' if the vector <hdL> is less than the vector <hdR>
**  and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
extern  Bag       LtVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  SumIntVector(<hdL>,<hdR>) . . . . . . . .  sum of an integer and a vector
**
**  'SumIntVector' returns the sum of the integer <hdL> and the vector <hdR>.
**  The  sum  is  a list,  where each  entry  is  the  sum of <hdL>  and  the
**  corresponding entry of <hdR>.
**
**  'SumIntVector' is an improved version  of  'SumSclList', which  does  not
**  call 'SUM' if the operands are immediate integers.
*/
extern  Bag       SumIntVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  SumVectorInt(<hdL>,<hdR>) . . . . . . . .  sum of a vector and an integer
**
**  'SumVectorInt' returns the sum of the vector <hdL> and the integer <hdR>.
**  The  sum  is  a list,  where each  entry  is  the  sum of <hdR>  and  the
**  corresponding entry of <hdL>.
**
**  'SumVectorInt' is an improved version  of  'SumListScl', which  does  not
**  call 'SUM' if the operands are immediate integers.
*/
extern  Bag       SumVectorInt (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  SumVectorVector(<hdL>,<hdR>)  . . . . . . . . . . . .  sum of two vectors
**
**  'SumVectorVector'  returns the sum  of the two  vectors <hdL>  and <hdR>.
**  The sum is  a new list, where each entry is the  sum of the corresponding
**  entries of <hdL> and <hdR>.
**
**  'SumVectorVector' is an improved version of 'SumListList', which does not
**  call 'SUM' if the operands are immediate integers.
*/
extern  Bag       SumVectorVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DiffIntVector(<hdL>,<hdR>)  . . . . difference of an integer and a vector
**
**  'DiffIntVector'  returns  the  difference of  the  integer <hdL> and  the
**  vector  <hdR>.   The difference  is  a list,  where  each  entry  is  the
**  difference of <hdL> and the corresponding entry of <hdR>.
**
**  'DiffIntVector'  is an  improved version of 'DiffSclList', which does not
**  call 'DIFF' if the operands are immediate integers.
*/
extern  Bag       DiffIntVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DiffVectorInt(<hdL>,<hdR>)  . . . . difference of a vector and an integer
**
**  'DiffVectorInt'  returns  the difference  of the  vector  <hdL>  and  the
**  integer <hdR>.   The difference  is  a  list,  where  each  entry is  the
**  difference of <hdR> and the corresponding entry of <hdL>.
**
**  'DiffVectorInt' is  an improved version of 'DiffListScl',  which does not
**  call 'DIFF' if the operands are immediate integers.
*/
extern  Bag       DiffVectorInt (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DiffVectorVector(<hdL>,<hdR>) . . . . . . . . . difference of two vectors
**
**  'DiffVectorVector' returns the difference of the  two vectors  <hdL>  and
**  <hdR>.  The difference is a  new list, where each entry is the difference
**  of the corresponding entries of <hdL> and <hdR>.
**
**  'DiffVectorVector' is an improved  version of  'DiffListList', which does
**  not call 'DIFF' if the operands are immediate integers.
*/
extern  Bag       DiffVectorVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdIntVector(<hdL>,<hdR>)  . . . . .  product of an integer and a vector
**
**  'ProdIntVector' returns the product of  the integer <hdL> and  the vector
**  <hdR>.  The product is the list, where each entry is the product of <hdL>
**  and the corresponding entry of <hdR>.
**
**  'ProdIntVector'  is an  improved version of 'ProdSclList', which does not
**  call 'PROD' if the operands are immediate integers.
*/
extern  Bag       ProdIntVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdVectorInt(<hdL>,<hdR>)  . . . . .  product of a scalar and an integer
**
**  'ProdVectorInt' returns the product of  the integer <hdR>  and the vector
**  <hdL>.  The product is the list, where each entry is the product of <hdR>
**  and the corresponding entry of <hdL>.
**
**  'ProdVectorInt'  is an  improved version of 'ProdSclList', which does not
**  call 'PROD' if the operands are immediate integers.
*/
extern  Bag       ProdVectorInt (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdVectorVector(<hdL>,<hdR>) . . . . . . . . . .  product of two vectors
**
**  'ProdVectorVector'  returns the  product  of the  two  vectors <hdL>  and
**  <hdR>.   The  product  is  the  sum of the products of the  corresponding
**  entries of the two lists.
**
**  'ProdVectorVector' is an improved version  of 'ProdListList',  which does
**  not call 'PROD' if the operands are immediate integers.
*/
extern  Bag       ProdVectorVector (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdVectorMatrix(<hdL>,<hdR>) . . . . .  product of a vector and a matrix
**
**  'ProdVectorMatrix' returns the product of the vector <hdL> and the matrix
**  <hdR>.  The product is the sum of the  rows  of <hdR>, each multiplied by
**  the corresponding entry of <hdL>.
**
**  'ProdVectorMatrix'  is an improved version of 'ProdListList',  which does
**  not  call 'PROD' and  also accummulates  the sum into  one  fixed  vector
**  instead of allocating a new for each product and sum.
*/
extern  Bag       ProdVectorMatrix (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  PowMatrixInt(<hdL>,<hdR>) . . . . . . .  power of a matrix and an integer
**
**  'PowMatrixInt' returns the <hdR>-th power of the matrix <hdL>, which must
**  be a square matrix of course.
**
**  Note that  this  function also  does the  inversion  of matrices when the
**  exponent is negative.
*/
extern  Bag       PowMatrixInt (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DepthVector( <hdVec> )  . . . . . . . . . . . . . . . . depth of a vector
*/
extern Bag DepthVector (
    Bag		hdVec );


/****************************************************************************
**
*F  InitVector()  . . . . . . . . . . . . . . . . . initialize vector package
**
**  'InitVector' initializes the vector package.
*/
extern  void            InitVector ( void );
