/****************************************************************************
**
*A  spiral_delay_list.c           SPIRAL source              Yevgen Voronenko
**
**
*/

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "list.h"                /* generic list package            */
#include        "range.h"               /* 'LEN_RANGE', 'LOW_RANGE', ..    */
#include        "record.h"              /* 'HdTilde', 'MakeRec'            */
#include        "namespaces.h"

#include        "list.h"
#include        "plist.h"

#include        "args.h"
#include        "spiral_delay_ev.h"

Obj HdNull;

#define TYPE_BAG_LIST(list) (GET_TYPE_BAG(INJECTION_D(list)))

/****************************************************************************
**
*F  LEN_BAG_LIST(<hdList>) . . . . . . . . . . . . . . .  length of a plain list
**
**  'LEN_BAG_LIST' returns   the logical length  of   the list  <hdList> as  a C
**  integer.   The length is stored  as GAP immediate  integer as the zeroeth
**  handle.
**
**  Note that 'LEN_BAG_LIST' is a  macro, so do  not call it with arguments that
**  have sideeffects.
**
**  'LEN_BAG_LIST'  is  defined  in  the declaration  part  of  this  package as
**  follows:
**
*/
UInt  LEN_BAG_LIST(Obj list) {
    Obj actual_list = GET_TYPE_BAG(list)==T_DELAY ? PTR_BAG(list)[0] : list;
    if ( GET_TYPE_BAG(actual_list) == T_INT || 
	 GET_TYPE_BAG(actual_list) == T_VAR || 
	 GET_TYPE_BAG(actual_list) == T_VARAUTO )
	return 0;
    else return NrHandles(GET_TYPE_BAG(actual_list), GET_SIZE_BAG(actual_list));
}

/****************************************************************************
**
*F  ELM_BAG_LIST(<hdList>,<pos>) . . . . . . . . . . . . element of a plain list
**
**  'ELM_BAG_LIST' return the <pos>-th element of the list <hdList>.  <pos> must
**  be a positive integer less than or equal to the length of <hdList>.
**
**  Note that  'ELM_BAG_LIST' is a macro, so do  not call it with arguments that
**  have sideeffects.
**
**  'ELM_BAG_LIST'  is  defined  in  the  declaration part  of  this  package as
**  follows:
**
*/
/* 0 based, unlike plain lists */
Obj  ELM_BAG_LIST(Obj list, UInt pos) {    
    Obj actual_list = GET_TYPE_BAG(list)==T_DELAY ? PTR_BAG(list)[0] : list;
    Obj res = PTR_BAG(actual_list)[pos-1];
    if ( res == 0 ) return HdNull;
    else return PROJECTION_D(res);
}

Obj  RAW_ELM_BAG_LIST(Obj list, UInt pos) {    
    Obj actual_list = GET_TYPE_BAG(list)==T_DELAY ? PTR_BAG(list)[0] : list;
    Obj res = PTR_BAG(actual_list)[pos-1];
    if ( res == 0 ) return HdNull;
    else return res; /* no projection */
}

/****************************************************************************
**
*F  SET_ELM_BAG_LIST(<hdList>,<pos>,<hdVal>) . assign an element to a plain list
**
**  'SET_ELM_BAG_LIST'  assigns the value <hdVal> to the plain list <hdList>  at
**  the position <pos>.  <pos> must be a positive integer  less than or equal
**  to the length of <hdList>.
**
**  Note that 'SET_ELM_BAG_LIST' is a  macro, so do not  call it  with arguments
**  that have sideeffects.
**
**  'SET_ELM_BAG_LIST' is defined  in the  declaration part  of this  package as
**  follows:
**
*/
/* 0 based, unlike plain lists */
void SET_ELM_BAG_LIST(Obj list, UInt pos, Obj val) {
    Obj actual_list = GET_TYPE_BAG(list)==T_DELAY ? PTR_BAG(list)[0] : list;
    SET_BAG(actual_list, pos-1,  INJECTION_D(val) );
}

/****************************************************************************
**
*F  LenBagList(<hdList>)  . . . . . . . . . . . . . . .  length of a plain list
**
**  'LenBagList' returns the length of the plain list <hdList> as a C integer.
**
**  'LenBagList' is the function in 'TabLenList' for plain lists.
*/
Int            LenBagList (Bag hdList)
{
    return LEN_BAG_LIST( hdList );
}


/****************************************************************************
**
*F  ElmBagList(<hdList>,<pos>)  . . . . . . . select an element of a plain list
**
**  'ElmBagList' selects  the  element at  position  <pos> of  the  plain  list
**  <hdList>.  It is the responsibility of the caller to ensure that <pos> is
**  a positive integer.   An error is signalled if <pos> is  larger than  the
**  length of <hdList> or if <hdList> has no  assigned value at the  position
**  <pos>.
**
**  'ElmfBagList' does the  same thing than 'ElmList', but need not check  that
**  <pos>  is  less  than or  equal to the  length  of <hdList>, this  is the
**  responsibility  of the  caller.   Also  it  returns 0 if  <hdList> has no
**  assigned value at the position <pos>.
**
**  'ElmBagList' is the function in 'TabElmList'  for plain lists.  'ElmfBagList'
**  is the function  in  'TabElmfList', 'TabElmlList', and  'TabElmrList' for
**  plain lists.
*/
Bag       ElmBagList (Bag hdList, Int pos)
{
    Bag           hdElm;          /* the selected element, result    */

    /* check the position                                                  */
    if ( LEN_BAG_LIST( hdList ) < pos ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos, 0 );
    }

    /* select and check the element                                        */
    hdElm = ELM_BAG_LIST( hdList, pos );
    if ( hdElm == 0 ) {
        return Error(
          "List Element: <list>[%d] must have a value",
                     pos, 0 );
    }

    /* return the element                                                  */
    return hdElm;
}

Bag       ElmfBagList (Bag hdList, Int pos)
{
    /* select and return the element                                       */
    return ELM_BAG_LIST( hdList, pos );
}


/****************************************************************************
**
*F  ElmsBagList(<hdList>,<hdPoss>)  . . . .  select a sublist from a plain list
**
**  'ElmsBagList'  returns a new list containing the elements at  the  position
**  given  in the list  <hdPoss> from  the  plain  list <hdList>.   It is the
**  responsibility  of  the  caller  to  ensure  that <hdPoss>  is  dense and
**  contains only positive integers.   An error is signalled if <hdList>  has
**  no assigned value at any of the positions in  <hdPoss>, or  if an element
**  of <hdPoss> is larger than the length of <hdList>.
**
**  'ElmsBagList' is the function in 'TabElmsList' for plain lists.
*/
Bag       ElmsBagList (Bag hdList, Bag hdPoss)
{
    Bag           hdElms;         /* selected sublist, result        */
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element from <list>         */
    Int                lenPoss;        /* length of <positions>           */
    Int                pos;            /* <position> as integer           */
    Int                i;              /* loop variable                   */

        /* get the length of <list>                                        */
        lenList = LEN_BAG_LIST( hdList );

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* make the result list                                            */
        hdElms = NewBag( T_LIST, SIZE_PLEN_PLIST( lenPoss ) );
        SET_LEN_PLIST( hdElms, lenPoss );

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
            if ( lenList < pos ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos, 0 );
            }

            /* select the element                                          */
            hdElm = ELM_BAG_LIST( hdList, pos );
            if ( hdElm == 0 ) {
                return Error(
                  "List Elements: <list>[%d] must have a value",
                             pos, 0 );
            }

            /* assign the element into <elms>                              */
            SET_ELM_PLIST( hdElms, i, hdElm );

        }

    /* return the result                                                   */
    return hdElms;
}


/****************************************************************************
**
*F  AssBagList(<hdList>,<pos>,<hdVal>)  . . . . . . . .  assign to a plain list
**
**  'AssBagList' assigns  the value <hdVal> to the plain list  <hdList> at  the
**  position <pos>.  It is the  responsibility of the caller to  ensure  that
**  <pos> is positive, and that <hdVal> is not 'HdVoid'.
**
**  If the position is larger then the length of the list <list>, the list is
**  automatically  extended.  To avoid  making this too often, the bag of the
**  list is extended by at least '<length>/8 + 4' handles.  Thus in the loop
**
**      l := [];  for i in [1..1024]  do l[i] := i^2;  od;
**
**  the list 'l' is extended only 32 times not 1024 times.
**
**  'AssBagList' is the function in 'TabAssList' for plain lists.
*/
Bag       AssBagList (Bag hdList, Int pos, Bag hdVal)
{

    /* resizing not supported                                              */
    if ( LEN_BAG_LIST( hdList ) < pos )
        return Error("Position must be in [%d, %d]", 1, LEN_BAG_LIST(hdList));

    /* now perform the assignment and return the assigned value            */
    SET_ELM_BAG_LIST( hdList, pos, hdVal );
    return hdVal;
}


/****************************************************************************
**
*F  AsssBagList(<hdList>,<hdPoss>,<hdVals>) . assign several elements to a list
**
**  'AsssBagList' assignes  the  values from the list <hdVals> at the positions
**  given  in  the  list   <hdPoss>   to  the   list  <hdList>.   It  is  the
**  responsibility of  the  caller  to ensure  that  <hdPoss>  is  dense  and
**  contains only positive integers, that <hdPoss> and <hdVals> have the same
**  length, and that <hdVals> is dense.
**
**  'AsssBagList' is the function in 'TabAsssList' for plain lists.
*/
Bag       AsssBagList (Bag hdList, Bag hdPoss, Bag hdVals)
{
    Int                lenPoss;        /* length of <positions>           */
    Int                pos;            /* <position> as integer           */
    Int                max;            /* larger position                 */
    Bag           hdVal;          /* one element from <vals>         */
    Int                i;              /* loop variable                   */

        /* get the length of <positions>                                   */
        lenPoss = LEN_LIST( hdPoss );

        /* find the largest entry in <positions>                           */
        max = LEN_BAG_LIST( hdList );
        for ( i = 1; i <= lenPoss; i++ ) {
            if ( max < HD_TO_INT( ELMF_LIST( hdPoss, i ) ) )
                max = HD_TO_INT( ELMF_LIST( hdPoss, i ) );
        }

	/* resizing not supported                                              */
	if ( LEN_BAG_LIST( hdList ) < max )
	    return Error("Positions must be in [%d, %d]", 1, LEN_BAG_LIST(hdList));

        /* loop over the entries of <positions> and select                 */
        for ( i = 1; i <= lenPoss; i++ ) {

            /* get <position>                                              */
            pos = HD_TO_INT( ELMF_LIST( hdPoss, i ) );

            /* select the element                                          */
            hdVal = ELMF_LIST( hdVals, i );

            /* assign the element into <elms>                              */
            SET_ELM_BAG_LIST( hdList, pos, hdVal );

        }

    /* return the result                                                   */
    return hdVals;
}


/****************************************************************************
**
*F  PosBagList(<hdList>,<hdVal>,<start>)  . .  position of an element in a list
**
**  'PosBagList'  returns the position of the value <hdVal> in  the plain  list
**  <hdList> after the first  position <start> as a C integer.  0 is returned
**  if <hdVal> is not in the list.
**
**  'PosBagList' is the function in 'TabPosList' for plain lists.
*/
Int            PosBagList (Bag hdList, Bag hdVal, Int start)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */
    Int                i;              /* loop variable                   */

    /* get the length of <list>                                            */
    lenList = LEN_BAG_LIST( hdList );
    hdVal = INJECTION_D(hdVal);

    /* loop over all entries in <list>                                     */
    for ( i = start+1; i <= lenList; i++ ) {

        /* select one element from <list>                                  */
        hdElm = RAW_ELM_BAG_LIST( hdList, i );

        /* compare with <val>                                              */
        if ( hdElm != 0 && (hdElm == hdVal || EQ( hdElm, hdVal ) == HdTrue) )
            break;

    }

    /* return the position (0 if <val> was not found)                      */
    return (lenList < i ? 0 : i);
}


/****************************************************************************
**
*F  PlainBagList(<hdList>)  . . . . . . .  convert a plain list to a plain list
**
**  'PlainBagList' converts the plain list <hdList> to a plain list.  Not  much
**  work.
**
**  'PlainBagList' is the function in 'TabPlainList' for plain lists.
*/
void            PlainBagList (Bag hdList)
{
    return;
}


/****************************************************************************
**
*F  IsDenseBagList(<hdList>)  . . . .  dense list test function for plain lists
**
**  'IsDenseBagList' returns 1 if the plain list <hdList> is a dense list and 0
**  otherwise.
**
**  'IsDenseBagList' is the function in 'TabIsDenseList' for plain lists.
*/
Int            IsDenseBagList (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Int                i;              /* loop variable                   */

    /* get the length of the list                                          */
    lenList = LEN_BAG_LIST( hdList );

    /* loop over the entries of the list                                   */
    for ( i = 1; i <= lenList; i++ ) {
        if ( RAW_ELM_BAG_LIST( hdList, i ) == 0 )
            return 0;
    }

    /* no hole found                                                       */
    return 1;
}


/****************************************************************************
**
*F  IsPossBagList(<hdList>) . . .  positions list test function for plain lists
**
**  'IsPossBagList' returns  1 if  the plain  list  <hdList>  is  a  dense list
**  containing only positive integers, and 0 otherwise.
**
**  'IsPossBagList' is the function in 'TabIsPossList' for plain lists.
*/
Int            IsPossBagList (Bag hdList)
{
    Int                lenList;        /* length of <list>                */
    Bag           hdElm;          /* one element of <list>           */
    Int                i;              /* loop variable                   */

    /* get the length of the variable                                      */
    lenList = LEN_BAG_LIST( hdList );

    /* loop over the entries of the list                                   */
    for ( i = 1; i <= lenList; i++ ) {
        hdElm = RAW_ELM_BAG_LIST( hdList, i );
        if ( hdElm == 0 || GET_TYPE_BAG(hdElm) != T_INT || HD_TO_INT(hdElm) <= 0 )
            return 0;
    }

    /* no problems found                                                   */
    return 1;
}


/****************************************************************************
**
*F  EqBagList(<hdL>,<hdR>) . . . . . . . . .  test if two plain lists are equal
**
**  'EqList' returns 'true' if the two plain lists <hdL> and <hdR> are  equal
**  and 'false' otherwise.
**
**  Is called from the 'EQ' binop so both  operands  are  already  evaluated.
*/
Bag       EqBagList (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    Int                i, tl, tr;

    tl = TYPE_BAG_LIST(hdL);
    tr = TYPE_BAG_LIST(hdR);
    if ( tl != tr )  
	return HdFalse;
    else if( tl == T_VAR )
	return EQ(INJECTION_D(hdL), INJECTION_D(hdR));

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_BAG_LIST( hdL );
    lenR = LEN_BAG_LIST( hdR );
    if ( lenL != lenR ) {
        return HdFalse;
    }

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL; i++ ) {
        hdElmL = RAW_ELM_BAG_LIST( hdL, i );
        hdElmR = RAW_ELM_BAG_LIST( hdR, i );
        if ( hdElmL == 0 && hdElmR != 0 ) {
            return HdFalse;
        }
        else if ( hdElmR == 0 && hdElmL != 0 ) {
            return HdFalse;
        }
        else if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return HdFalse;
        }
    }

    /* no differences found, the lists are equal                           */
    return HdTrue;
}


/****************************************************************************
**
*F  LtBagList(<hdL>,<hdR>)  . . . . . . . . . test if two plain lists are equal
**
**  'LtList' returns 'true' if  the  plain list <hdL>  is less than the plain
**  list <hdR> and 'false' otherwise.
**
**  Is called from the 'LT' binop so both operands are already evaluated.
*/
Bag       LtBagList (Bag hdL, Bag hdR)
{
    Int                lenL;           /* length of the left operand      */
    Int                lenR;           /* length of the right operand     */
    Bag           hdElmL;         /* element of the left operand     */
    Bag           hdElmR;         /* element of the right operand    */
    Int                i, tl, tr;
    
    tl = TYPE_BAG_LIST(hdL);
    tr = TYPE_BAG_LIST(hdR);
    if ( tl != tr )  
	return (tl < tr) ? HdTrue : HdFalse;
    else if( tl == T_VAR )
	return LT(INJECTION_D(hdL), INJECTION_D(hdR));

    /* get the lengths of the lists and compare them                       */
    lenL = LEN_BAG_LIST( hdL );
    lenR = LEN_BAG_LIST( hdR );

    /* loop over the elements and compare them                             */
    for ( i = 1; i <= lenL && i <= lenR; i++ ) {
        hdElmL = RAW_ELM_BAG_LIST( hdL, i );
        hdElmR = RAW_ELM_BAG_LIST( hdR, i );
        if ( hdElmL == 0 && hdElmR != 0 ) {
            return HdTrue;
        }
        else if ( hdElmR == 0 && hdElmL != 0 ) {
            return HdFalse;
        }

        else if ( hdElmL != hdElmR && EQ( hdElmL, hdElmR ) == HdFalse ) {
            return LT( hdElmL, hdElmR );
        }
    }

    /* reached the end of at least one list                                */
    return (lenL < lenR) ? HdTrue : HdFalse;
}


Obj  FunBagList ( Obj hdCall ) {
    char * usage = "usage: BagList( <type>, <list> )";
    Obj   hdT, hdLst, hdRes;
    UInt  typ, i, nelms;

    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 3 * SIZE_HD )  return Error(usage, 0,0);

    hdT = EVAL(PTR_BAG(hdCall)[1]); 
    if ( GET_TYPE_BAG(hdT) != T_INT ) return Error(usage, 0, 0);
    hdLst = EVAL(PTR_BAG(hdCall)[2]); 
    if ( ! IS_LIST(hdLst) ) return Error(usage, 0, 0);

    typ = HD_TO_INT(hdT);
    nelms = LEN_LIST(hdLst);

    hdRes = NewBag(typ, nelms * SIZE_HD);
    for ( i = 1; i <= nelms; ++i ) {
	SET_BAG(hdRes, i-1,  INJECTION_D(ELM_LIST(hdLst, i)) );
    }
    return PROJECTION_D(hdRes);
}

/****************************************************************************
**
*F  InitBagList() . . . . . . . . . . . . . . . . . initialize the list package
**
**  Is called during  the  initialization  to  initialize  the  list package.
*/
void            InitSPIRAL_BagList (void)
{
    UInt T;

#define INSTALL_FUNCS(T) \
        if ( T != T_REC) TabIsList[T]       = 1; \
	TabLenList[T]      = LenBagList; \
	TabElmList[T]      = ElmBagList; \
	TabElmfList[T]     = ElmfBagList; \
	TabElmlList[T]     = ElmfBagList; \
	TabElmrList[T]     = ElmfBagList; \
	TabElmsList[T]     = ElmsBagList; \
	TabAssList[T]      = AssBagList; \
	TabAsssList[T]     = AsssBagList; \
	TabPosList[T]      = PosBagList; \
	TabPlainList[T]    = PlainBagList; \
	TabIsDenseList[T]  = IsDenseBagList; \
	TabIsPossList[T]   = IsPossBagList; \
	if ( T != T_VAR && T != T_REC ) { \
	  TabEq[T][T]        = EqBagList; \
	  TabLt[T][T]        = LtBagList; \
	}

    for(T = T_DELAY; T < LIST_TAB_SIZE; ++T) {
	if ( T != T_VAR ) {
	    INSTALL_FUNCS(T);
	}
    }
    INSTALL_FUNCS(T_REC);

#undef INSTALL_FUNCS
    GlobalPackage2("spiral", "delay");
    InstIntFunc("BagList", FunBagList);
    EndPackage();

    InitGlobalBag(&HdNull, "HdNull");
    HdNull = StringToHd("null");
}


/****************************************************************************
**
*E  Emacs . . . . . . . . . . . . . . . . . . . . . . . local emacs variables
**
**  Local Variables:
**  c-basic-offset:     4
**  outline-regexp:     "*A\\|*F\\|*V\\|*T\\|*E"
**  fill-column:        73
**  fill-prefix:        "**  "
**  End:
*/
