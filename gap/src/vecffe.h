/****************************************************************************
**
*A  vecffe.h                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file  declares  the functions that mainly  operate on vectors  whose
**  elements are elements from  finite fields.  As vectors  are special lists
**  many things are done in the list package.
**
**  A *vector* is a list that has no holes,  and whose elements all come from
**  a common field.  For the full definition of vectors see chapter "Vectors"
**  in  the {\GAP} manual.   Read also about "More   about Vectors" about the
**  vector flag and the compact representation of vectors over finite fields.
**
**  This package consists of four parts.
**
**  The   first   part    consists   of   the   macros    'SIZE_PLEN_VECFFE',
**  'PLEN_SIZE_VECFFE',   'LEN_VECFFE',    'SET_LEN_VECFFE',    'VAL_VECFFE',
**  'SET_VAL_VECFFE', 'ELM_VECFFE', and 'SET_ELM_VECFFE'.  They determine the
**  representation of vectors.  Everything else in  this file and the rest of
**  the {\GAP} kernel uses those macros to access and modify vectors.
**
**  The  second  part  consists of  the functions  'LenVecFFE',  'ElmVecFFE',
**  'ElmsVecFFE',  AssVecFFE', 'AsssVecFFE',  'PlainVecFFE', 'IsDenseVecFFE',
**  and 'IsPossVecFFE'.  They are the functions required by the generic lists
**  package.  Using these functions  the other parts of the {\GAP} kernel can
**  access  and  modify vectors  without actually being  aware  that they are
**  dealing with a vector.
**
**  The   third   part    consists   of   the   functions    'IsXTypeVecFFE',
**  'IsXTypeMatFFE', 'SumFFEVecFFE', 'SumVecFFEFFE', 'SumVecFFEVecFFE',  etc.
**  They  are the function for binary operations,  which  overlay the generic
**  functions in the generic list package for better efficiency.
**
**  The fourth part contains the internal functions for vectors.
**
*/


/****************************************************************************
**
*F  PLEN_SIZE_VECFFE(<size>)  . . . .  physical length from size for a vector
**
**  'PLEN_SIZE_VECFFE' computes  the  physical length  (e.g.   the  number of
**  elements that could be stored in a list) from the <size> (as reported  by
**  'GET_SIZE_BAG') for a vector.
**
**  Note that 'PLEN_SIZE_VECFFE' is a macro, so do not call it with arguments
**  that have sideeffects.
*/
#define PLEN_SIZE_VECFFE(GET_SIZE_BAG)          (((GET_SIZE_BAG) - SIZE_HD) / sizeof(TypFFE))


/****************************************************************************
**
*F  SIZE_PLEN_VECFFE(<plen>)  .  size for a vector with given physical length
**
**  'SIZE_PLEN_VECFFE' returns  the  size that a  vector with room for <plen>
**  elements must at least have.
**
**  Note that 'SIZE_PLEN_VECFFE' is a macro, so do not call it with arguments
**  that have sideeffects.
*/
#define SIZE_PLEN_VECFFE(PLEN)          (SIZE_HD + (PLEN) * sizeof(TypFFE))


/****************************************************************************
**
*F  LEN_VECFFE(<hdList>)  . . . . . . . . . . . . . . . .  length of a vector
**
**  'LEN_VECFFE' returns the  logical  length  of  the vector <hdList> as a C
**  integer.  The  length  is stored as GAP immediate integer as  the zeroeth
**  handle.
**
**  Note that 'LEN_VECFFE' is a macro,  so do not call it with arguments that
**  have sideeffects.
*/
#define LEN_VECFFE(LIST)                PLEN_SIZE_VECFFE( GET_SIZE_BAG( LIST ) )


/****************************************************************************
**
*F  SET_LEN_VECFFE(<hdList>,<len>)  . . . . . . .  set the length of a vector
**
**  'SET_LEN_VECFFE' sets  the  length of the vector <hdList> to <len>.   The
**  length is stored as GAP immediate integer as the zeroeth handle.
**
**  Note that 'SET_LEN_VECFFE' is a macro, so do not call  it  with arguments
**  that have sideeffects.
*/
#define SET_LEN_VECFFE(LIST,LEN)        Resize( LIST, SIZE_PLEN_VECFFE(LEN) )


/****************************************************************************
**
*F  FLD_VECFFE(<hdList>)  . . . . . . . . . . . . . . . . . field of a vector
**
**  'FLD_VECFFE' returns the handle of the finite field over which the vector
**  <hdList> is defined.
**
**  Note that 'FLD_VECFFE' is a macro, so do not call it  with arguments that
**  have sideeffects.
*/
#define FLD_VECFFE(LIST)                (PTR_BAG(LIST)[0])


/****************************************************************************
**
*F  SET_FLD_VECFFE(<hdList>,<hdField>)  . . . . . . set the field of a vector
**
**  'SET_FLD_VECFFE' sets the field of the vector <hdList> to <hdField>.
**
**  Note that 'SET_FLD_VECFFE' is a macro, so  do not  call it with arguments
**  that have sideeffects.
*/
#define SET_FLD_VECFFE(LIST,FLD)        SET_BAG(LIST, 0, (FLD))


/****************************************************************************
**
*F  VAL_VECFFE(<hdVec>,<pos>) . . . . . . . value of an element from a vector
**
**  'VAL_VECFFE' returns the value of the  <pos>-th  element  of  the  finite
**  field vector <hdVec>.
**
**  Note that 'VAL_VECFFE' is a macro, so do not call it with arguments  that
**  have sideeffects.
*/
#define VAL_VECFFE(VEC,POS)             (((TypFFE*)(PTR_BAG(VEC)+1))[(POS)-1])


/****************************************************************************
**
*F  SET_VAL_VECFFE(<hdVec>,<pos>,<val>) set value of an element from a vector
**
**  'SET_VAL_VECFFE' sets the value of the <pos>-th  element  of  the  finite
**  field vector <hdVec> to <val>.
**
**  Note that 'SET_VAL_VECFFE' is a macro, so do not call it  with  arguments
**  that have sideeffects.
*/
#define SET_VAL_VECFFE(VEC,POS,VAL)     (VAL_VECFFE(VEC,POS) = (VAL))


/****************************************************************************
**
*F  ELM_VECFFE(<hdVec>,<i>,<hdElm>) . . . .  element of a finite field vector
**
**  'ELM_VECFFE'  assigns the   <i>-th  element of  the finite   field vector
**  <hdVec> to the finite field element bag <hdElm>.   <i> must be a positive
**  integer less than or equal to the length of <hdVec>.
**
**  Note that 'ELM_VECFFE' is a macro, so do not call  it with arguments that
**  have sideeffects.
**
**  'ELM_VECFFE' is one of the functions that packages implementing list like
**  objects must export.  It is called from 'ElmList' and 'EvFor' and various
**  other places.  Note that 'ELM_VECFFE' expects  the  bag  for  the  result
**  already allocated, unlike all the other 'ELM_<type>' functions.
*/
#define ELM_VECFFE(LIST,POS,ELM)      (SET_FLD_FFE(ELM,FLD_VECFFE(LIST)), \
                                       SET_VAL_FFE(ELM,VAL_VECFFE(LIST,POS)))


/****************************************************************************
**
*F  SET_ELM_VECFFE(<hdList>,<pos>,<hdVal>)  . . assign an element to a vector
**
**  'SET_ELM_VECFFE' assigns the value <hdVal> to the  vector <hdList> at the
**  position <pos>.  <pos> must be a positive integer less  than  or equal to
**  the length of <hdList>.
**
**  Note that 'SET_ELM_VECFFE'  is a macro, so do not call it  with arguments
**  that have sideeffects.
*/
#define SET_ELM_VECFFE(LIST,POS,ELM)    SET_VAL_VECFFE(LIST,POS,VAL_FFE(ELM))


/****************************************************************************
**
*F  LenVecFFE(<hdList>) . . . . . . . . . . . . . . . . .  length of a vector
**
**  'LenVecFFE' returns the length of the vector <hdList> as a C integer.
**
**  'LenVecFFE' is the function in 'TabLenList' for vectors.
*/
extern  Int            LenVecFFE (
            Bag           hdList );


/****************************************************************************
**
*F  ElmVecFFE(<hdList>,<pos>) . . . . . . . . . select an element of a vector
**
**  'ElmVecFFE' selects the element at position <pos> of the vector <hdList>.
**  It is the responsibility of the caller to ensure that <pos> is a positive
**  integer.  An  error is signalled if <pos>  is  larger than the  length of
**  <hdList>.
**
**  'ElmfVecFFE' does  the same thing than 'ElmList', but need not check that
**  <pos>  is less than  or  equal to the  length of  <hdList>, this  is  the
**  responsibility of the caller.
**
**  'ElmVecFFE' is the function in 'TabElmList' for vectors.  'ElmfVecFFE' is
**  the  function  in  'TabElmfList', 'TabElmlList',  and  'TabElmrList'  for
**  vectors.
*/
extern  Bag       ElmVecFFE (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmfVecFFE (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmlVecFFE (
            Bag           hdList,
            Int                pos );

extern  Bag       ElmrVecFFE (
            Bag           hdList,
            Int                pos );


/****************************************************************************
**
*F  ElmsVecFFE(<hdList>,<hdPoss>) . . . . . .  select a sublist from a vector
**
**  'ElmsVecFFE' returns a new list containing the elements  at the  position
**  given  in  the  list  <hdPoss>  from  the  vector  <hdList>.  It  is  the
**  responsibility  of  the  caller  to  ensure that  <hdPoss> is  dense  and
**  contains only positive integers.   An error is signalled if an element of
**  <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsVecFFE' is the function in 'TabElmsList' for vectors.
*/
extern  Bag       ElmsVecFFE (
            Bag           hdList,
            Bag           hdPoss );


/****************************************************************************
**
*F  AssVecFFE(<hdList>,<pos>,<hdVal>) . . . . . . . . . .  assign to a vector
**
**  'AssVecFFE' assigns the  value  <hdVal>  to the  vector  <hdList>  at the
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
**  'AssVecFFE' is the function in 'TabAssList' for vectors.
**
**  'AssVecFFE' simply converts  the  vector into a plain list, and then does
**  the same  stuff as  'AssPlist'.   This is because  a  vector is  not very
**  likely to stay a vector after the assignment.
*/
extern  Bag       AssVecFFE (
            Bag           hdList,
            Int                pos,
            Bag           hdVal );


/****************************************************************************
**
*F  AsssVecFFE(<hdList>,<hdPoss>,<hdVals>)assign several elements to a vector
**
**  'AsssVecFFE' assignes the values from the  list <hdVals> at the positions
**  given  in  the  list  <hdPoss>  to   the  vector  <hdList>.   It  is  the
**  responsibility  of the  caller to  ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssVecFFE' is the function in 'TabAsssList' for vectors.
**
**  'AsssVecFFE' simply converts the vector to a plain list and then does the
**  same stuff as 'AsssPlist'.  This is because a vector  is not  very likely
**  to stay a vector after the assignment.
*/
extern  Bag       AsssVecFFE (
            Bag           hdList,
            Bag           hdPoss,
            Bag           hdVals );


/****************************************************************************
**
*F  PosVecFFE(<hdList>,<hdVal>,<pos>) . .  position of an element in a vector
**
**  'PosVecFFE' returns  the  position  of the  value  <hdVal>  in the vector
**  <hdList> after the first position <start> as a C  integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosVecFFE' is the function in 'TabPosList' for vectors.
*/
extern  Int            PosVecFFE (
            Bag           hdList,
            Bag           hdVal,
            Int                start );


/****************************************************************************
**
*F  PlainVecFFE(<hdList>) . . . . . . . . .  convert a vector to a plain list
**
**  'PlainVecFFE'  converts the vector  <hdList> to  a plain list.   Not much
**  work.
**
**  'PlainVecFFE' is the function in 'TabPlainList' for vectors.
*/
extern  void            PlainVecFFE (
            Bag           hdList );


/****************************************************************************
**
*F  IsDenseVecFFE(<hdList>) . . . . . .  dense list test function for vectors
**
**  'IsDenseVecFFE' returns 1, since every vector is dense.
**
**  'IsDenseVecFFE' is the function in 'TabIsDenseList' for vectors.
*/
extern  Int            IsDenseVecFFE (
            Bag           hdList );


/****************************************************************************
**
*F  IsPossVecFFE(<hdList>)  . . . .  positions list test function for vectors
**
**  'IsPossVecFFE' returns 0, since every vector contains no integers.
**
**  'IsPossVecFFE' is the function in 'TabIsPossList' for vectors.
*/
extern  Int            IsPossVecFFE (
            Bag           hdList );


/****************************************************************************
**
*F  IsXTypeVecFFE(<hdList>) . . . . . . . . . . .  test if a list is a vector
**
**  'IsXTypeVecFFE'  returns 1  if  the list  <hdList>  is  a  vector  and  0
**  otherwise.  As a sideeffect the representation  of the list is changed to
**  'T_VECFFE'.
**
**  'IsXTypeVecFFE' is the function in 'TabIsXTypeList' for vectors.
*/
extern  Int            IsXTypeVecFFE (
            Bag           hdList );


/****************************************************************************
**
*F  IsXTypeMatFFE(<hdList>) . . . . . . . . . . .  test if a list is a matrix
**
**  'IsXTypeMatFFE'  returns  1  if  the list <hdList>  is  a  matrix  and  0
**  otherwise.  As a sideeffect the representation  of the rows is changed to
**  'T_VECFFE'.
**
**  'IsXTypeMatFFE' is the function in 'TabIsXTypeList' for matrices.
*/
extern  Int            IsXTypeMatFFE (
            Bag           hdList );


/****************************************************************************
**
*F  SumFFEVecFFE(<hdL>,<hdR>) . .  sum of a finite field element and a vector
**
**  'SumFFEVecFFE' returns  the sum of the finite field element <hdL> and the
**  vector <hdR>.  The sum is a  new list, where each element is  the sum  of
**  <hdL> and the corresponding entry of <hdR>.
**
**  'SumFFEVecFFE'  is an improved version of  'SumSclList', which  does  not
**  call 'SUM'.
*/
extern  Bag       SumFFEVecFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  SumVecFFEFFE(<hdL>,<hdR>) . .  sum of a vector and a finite field element
**
**  'SumVecFFEFFE' returns the  sum  of the vector <hdL> and the finite field
**  element  <hdR>.  The sum is a new list,  where each element is the sum of
**  <hdL> and the corresponding entry of <hdR>.
**
**  'SumVecFFEFFE'  is an improved  version of 'SumListScl',  which does  not
**  call 'SUM'.
*/
extern  Bag       SumVecFFEFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  SumVecFFEVecFFE(<hdL>,<hdR>)  . . . . . . . . . . . .  sum of two vectors
**
**  'SumVecFFEVecFFE' returns  the  sum of  the two vectors <hdL>  and <hdR>.
**  The sum is a new list, where each element is the sum of the corresponding
**  entries of <hdL> and <hdR>.
**
**  'SumVecFFEVecFFE' is an improved version of 'SumListList', which does not
**  call 'SUM'.
*/
extern  Bag       SumVecFFEVecFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DiffFFEVecFFE(<hdL>,<hdR>)   diff. of a finite field element and a vector
**
**  'DiffFFEVecFFE' returns the  difference of the finite field element <hdL>
**  and the  vector <hdR>.  The difference is a new  list, where each element
**  is the difference of <hdL> and the corresponding entry of <hdR>.
**
**  'DiffFFEVecFFE' is an improved version  of 'DiffSclList', which  does not
**  call 'DIFF'.
*/
extern  Bag       DiffFFEVecFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DiffVecFFEFFE(<hdL>,<hdR>)   diff. of a vector and a finite field element
**
**  'DiffVecFFEFFE' returns the difference of the vector <hdL> and the finite
**  field element <hdR>.  The difference is a new list, where each element is
**  the difference of <hdL> and the corresponding entry of <hdR>.
**
**  'DiffVecFFEFFE'  is an  improved version of 'DiffListScl', which does not
**  call 'DIFF'.
*/
extern  Bag       DiffVecFFEFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  DiffVecFFEVecFFE(<hdL>,<hdR>) . . . . . . . . . difference of two vectors
**
**  'DiffVecFFEVecFFE' returns the difference of the  two  vectors  <hdL> and
**  <hdR>.   The  difference  is  a  new  list,  where  each  element is  the
**  difference of the corresponding entries of <hdL> and <hdR>.
**
**  'DiffVecFFEVecFFE' is an  improved version of 'DiffListList',  which does
**  not call 'PROD'.
*/
extern  Bag       DiffVecFFEVecFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdFFEVecFFE(<hdL>,<hdR>) product of a finite field element and a vector
**
**  'ProdFFEVecFFE' returns the product of the finite field element <hdL> and
**  the vector <hdR>.  The product is  a new list,  where each element is the
**  product of <hdL> and the corresponding entry of <hdR>.
**
**  'ProdFFEVecFFE' is an improved version  of 'ProdSclList', which  does not
**  call 'PROD'.
*/
extern  Bag       ProdFFEVecFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdVecFFEFFE(<hdL>,<hdR>) product of a vector and a finite field element
**
**  'ProdVecFFEFFE' returns  the  product of the vector <hdL>  and the finite
**  field element <hdR>.   The product  is a new list,  where each element is
**  the product of <hdL> and the corresponding entry of <hdR>.
**
**  'ProdVecFFEFFE'  is an  improved version of 'ProdListScl', which does not
**  call 'PROD'.
*/
extern  Bag       ProdVecFFEFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdVecFFEVecFFE(<hdL>,<hdR>) . . . . . . . . . .  product of two vectors
**
**  'ProdVecFFEVecFFE'  returns  the product  of the  two  vectors <hdL>  and
**  <hdR>.  The product is the sum of the corresponding entries of <hdL>  and
**  <hdR>.
**
**  'ProdVecFFEVecFFE' is an improved version  of 'ProdListList',  which does
**  not call 'PROD'.
*/
extern  Bag       ProdVecFFEVecFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  ProdVecFFEMatFFE(<hdL>,<hdR>) . . . . .  product of a vector and a matrix
**
**  'ProdVecFFEMatFFE' returns the product of the vector <hdL> and the matrix
**  <hdR>.  The product is the sum of the  rows of <hdR>, each  multiplied by
**  the corresponding entry of <hdL>.
**
**  'ProdVecFFEMatFFE'  is  an improved version of 'ProdListList', which does
**  not  call  'PROD'  and also  acummulates the sum  into  one fixed  vector
**  instead of allocating a new for each product and sum.
*/
extern  Bag       ProdVecFFEMatFFE (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  PowMatFFEInt(<hdL>,<hdR>) . . . . . . .  power of a matrix and an integer
**
**  'PowMatFFEInt' returns the <hdR>-th power of the matrix <hdL>, which must
**  be a square matrix of course.
**
**  Note that  this  function also  does the  inversion  of matrices when the
**  exponent is negative.
*/
extern  Bag       PowMatFFEInt (
            Bag           hdL,
            Bag           hdR );


/****************************************************************************
**
*F  PrVecFFE(<hdList>)  . . . . . . . . . . . . . . . . . . .  print a vector
**
**  'PrVecFFE' prints a vector.
*/
extern  void            PrVecFFE (
            Bag           hdList );


/****************************************************************************
**
*F  DepthVecFFE( <hdVec> )  . . . . . . . . .  depth of a finite field vector
*/
extern Bag DepthVecFFE ( 
		Bag	hdVec );


/****************************************************************************
**
*F  CharVecFFE(<hdVec>) . . . . . . . . . . . . .  characteristic of a vector
**
**  'CharVecFFE' returns  the  characteristic  of  the  field  in  which  the
**  elements of the finite field vector <hdVec> lie.
*/
extern  Int            CharVecFFE (
            Bag           hdVec );


/****************************************************************************
**
*F  CharMatFFE(<hdMat>) . . . . . . . . . . . . .  characteristic of a matrix
**
**  'CharMatFFE'  returns  the  characteristic of  the  field  in  which  the
**  elements of the finite field matrix <hdMat> lie.
*/
extern  Int            CharMatFFE (
            Bag           hdMat );


/****************************************************************************
**
*F  FunCharFFE( <hdCall> )  . . . .  characteristic of a finite field element
**
**  'FunCharFFE' implements the internal function 'CharFFE'.
**
**  'CharFFE( <z> )' or 'CharFFE( <vec> )' or 'CharFFE( <mat> )'
**
**  'CharFFE' returns the  characteristic of the   smallest finite field  <F>
**  containing the element <z>, respectively all elements of the vector <vec>
**  over a finite field (see "Vectors"), or matrix  <mat> over a finite field
**  (see "Matrices").
*/
extern  Bag       FunCharFFE (
            Bag           hdCall );


/****************************************************************************
**
*F  DegreeVecFFE(<hdVec>) . . . . . . . . . . . . . . . .  degree of a vector
**
**  'DegreeVecFFE' returns  the  degree  of the  smallest finite  field  that
**  contains all elements of the finite field vector <hdVec>.
*/
extern  Int            DegreeVecFFE (
            Bag           hdVec );


/****************************************************************************
**
*F  DegreeMatFFE(<hdMat>) . . . . . . . . . . . . . . . .  degree of a matrix
**
**  'DegreeMatFFE' returns  the  degree  of the  smallest finite  field  that
**  contains all elements of the finite field matrix <hdMat>.
*/
extern  Int            DegreeMatFFE (
            Bag           hdMat );


/****************************************************************************
**
*F  FunDegreeFFE( <hdCall> )  . . . . . . .  degree of a finite field element
**
**  'FunDegreeFFE' implements the internal function 'DegreeFFE'.
**
**  'DegreeFFE( <z> )' or 'DegreeFFE( <vec> )' or 'DegreeFFE( <mat> )'
**
**  'DegreeFFE'   returns the   degree   of  the  smallest  finite  field <F>
**  containing the element <z>, respectively all elements of the vector <vec>
**  over a finite field (see "Vectors"), or matrix <mat> over a  finite field
**  (see  "Matrices").  For vectors and matrices,  an error  is raised if the
**  smallest finite field containing all elements of the vector or matrix has
**  order larger than $2^{16}$.
*/
extern  Bag       FunDegreeFFE (
            Bag           hdCall );


/****************************************************************************
**
*F  FunLogVecFFE( <hdCall> )  . . . . . . . . . internal function 'LogVecFFE'
**
**  'FunLogVecFFE' implements the internal function 'LogVecFFE'.
**
**  'LogVecFFE( <vector>, <position> )'
*/
extern  Bag       FunLogVecFFE (
            Bag           hdCall );


/****************************************************************************
**
*F  FunMakeVecFFE( <hdCall> ) . . . . . . . .  internal function 'MakeVecFFE'
**
**  'FunMakeVecFFE' implements the internal function 'MakeVecFFE'.
**
**  'MakeVecFFE( <list>, <ffe> )'
*/
extern  Bag       FunMakeVecFFE (
            Bag           hdCall );


/****************************************************************************
**
*F  FunNumberVecFFE( <hdCall> ) . . . . . .  internal function 'NumberVecFFE'
**
**  'FunNumberVecFFE' implements the internal function 'NumberVecFFE'.
**
**  'NumberVecFFE( <vector>, <powers>, <integers> )'
*/
extern  Bag       FunNumberVecFFE (
            Bag           hdCall );


/****************************************************************************
**
*F  FunDepthVector( <hdCall> )  . . . . . . . internal function 'DepthVector'
*/
extern  Bag       FunDepthVector (
            Bag           hdCall );


/****************************************************************************
**
*F  InitVecFFE()  . . . . . . . . . . . . . . . . . initialize vector package
**
**  'InitVecFFE' initializes the finite field vector package.
*/
extern  void            InitVecFFE ( void );
