/****************************************************************************
**
*A  record.c                    GAP source                   Martin Schoenert
**
**
*Y  Copyright (C) 2018-2021, Carnegie Mellon University
*Y  All rights reserved.  See LICENSE for details.
*Y  
*Y  This work is based on GAP version 3, with some files from version 4.  GAP is
*Y  Copyright (C) (1987--2021) by the GAP Group (www.gap-system.org).
**
**  This file contains the functions for computing with records.
**
**  A record with <n> elements is stored as a bag with 2  * <n> entries.  The
**  odd entries are the record names of the elements and the even entries are
**  the corresponding values.
**
**
*N  05-Jun-90 martin 'PrRec' should be capable of ignoring elements
*N  05-Jun-90 martin 'PrRec' should support '~.<path>'
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage management      */
#include        "scanner.h"             /* reading of tokens and printing  */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "idents.h"              /* 'FindRecname'                   */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "gstring.h"              /* 'IsString'                      */

#include        "record.h"              /* declaration part of the package */
#include        "tables.h"              /* TableLookup                     */
#include        "args.h"                /* StringToHd                      */
#include        "debug.h"               /* debugging */
#include        "comments.h"            /* HdDocRecname */
#include		"GapUtils.h"

/****************************************************************************
**
*V  HdRnOp  . . . . . . . . handle of the 'operations' record name bag, local
*V  HdBases . . . . . . . . handle of the '__bases__'  record name bag, local
*V  HdCall1 . . . . . . . . . handle of a function call bag with 1 arg, local
*V  HdCall2 . . . . . . . . . handle of a function call bag with 2 arg, local
*/
Bag       HdRnOp;
Bag       HdBases;
Bag       HdCall1,  HdCall2;


/* Find a field in a record without looking in __bases__,
   returns 0 if not found */
Obj*   _FindRecnameRec_nobases ( Obj hdRec, Obj hdField ) {
    Obj * ptRec, * ptEnd;
    ptRec = (Obj*)PTR_BAG(hdRec);
    ptEnd = (Obj*)((char*)ptRec + GET_SIZE_BAG(hdRec));
    while ( ptRec < ptEnd && ptRec[0] != hdField )  ptRec += 2;

    /* was it found ?                                                      */
    if ( ptRec == ptEnd ) return 0;
    else return ptRec;
}

/* Find a field in a record, look in __bases__ list if not found,
   returns 0 if not found neither in record nor in __bases__,
   returns bag which contains obj record in *hdBag.
   FindRecnameRecWithDepth() is a copy of this function for performance
   reasons only.
   
Obj*   FindRecnameRec ( Obj hdRec, Obj hdField, Obj* phdBag ) {
    Int depth = 0;
    return FindRecnameRecWithDepth(hdRec, hdField, phdBag, &depth);
}

*/
   
Obj*   FindRecnameRec ( Obj hdRec, Obj hdField, Obj* phdBag ) {
    Obj * ptRec = _FindRecnameRec_nobases(hdRec, hdField);
    if ( ptRec == 0 ) {
        Bag * ptBases = _FindRecnameRec_nobases(hdRec, HdBases);
        if ( ptBases != 0 && ptBases[1] != 0) {
            int i;
            Obj bases = ptBases[1];
            if ( GET_TYPE_BAG(bases) != T_LIST && GET_TYPE_BAG(bases) != T_VECTOR ) {
                Error("FindRecnameRec: __bases__ field of <%g> must be a plain list",
                      (Int)hdRec, 0);
                *phdBag = 0;
                return 0;
            }
            for ( i = 1; i <= LEN_PLIST(bases); ++i) {
                ptRec = FindRecnameRec( ELM_PLIST(bases, i), hdField, phdBag);
                if ( ptRec != 0 ) return ptRec; /* found in one of __bases__ */
            }
        }
        *phdBag = 0;
        return 0; /* nothing found */
    }
    else { *phdBag = hdRec; return ptRec; } /* found directly in record */
}

Obj*   FindRecnameRecWithDepth ( Obj hdRec, Obj hdField, Obj* phdBag, Int* depth ) {
    Obj * ptRec = _FindRecnameRec_nobases(hdRec, hdField);
    if ( *depth>0 || ptRec == 0 ) {
        Bag * ptBases = _FindRecnameRec_nobases(hdRec, HdBases);
        if (ptRec != 0) (*depth)--;
        if ( ptBases != 0 && ptBases[1] != 0) {
            int i;
            Obj bases = ptBases[1];
            if ( GET_TYPE_BAG(bases) != T_LIST && GET_TYPE_BAG(bases) != T_VECTOR ) {
                Error("FindRecnameRec: __bases__ field of <%g> must be a plain list",
                      (Int)hdRec, 0);
                *phdBag = 0;
                return 0;
            }
            for ( i = 1; i <= LEN_PLIST(bases); ++i) {
                ptRec = FindRecnameRecWithDepth( ELM_PLIST(bases, i), hdField, phdBag, depth);
                if ( ptRec != 0 ) return ptRec; /* found in one of __bases__ */
            }
        }
        *phdBag = 0;
        return 0; /* nothing found */
    }
    else { *phdBag = hdRec; return ptRec; } /* found directly in record */
}

/* Sets the field value in a record, if needed a field is added to a record,
   by allocating additional storage */
Obj    SetRecname ( Obj hdRec, Obj hdField, Obj hdVal ) {
    Obj  * ptRec;
    Obj  hdRealRec;
    if ( GET_TYPE_BAG(hdRec) != T_REC )
        return Error("SetRecname: Record expected", 0, 0);
    ptRec = FindRecnameRec(hdRec, hdField, &hdRealRec);
    if ( ptRec != 0 ) { /* field exists */
        ptRec[1] = hdVal;
        //  CHANGED_BAG(hdRealRec);
    } else { /* create a new field */
        Resize(hdRec, GET_SIZE_BAG(hdRec) + 2*SIZE_HD);
        SET_BAG(hdRec,  GET_SIZE_BAG(hdRec) / SIZE_HD - 2 ,  hdField );
        SET_BAG(hdRec,  GET_SIZE_BAG(hdRec) / SIZE_HD - 1 ,  hdVal );
    }
    return hdVal;
}

/* Sets the field value in a record, if needed a field is added to a record,
   by allocating additional storage
   (same as SetRecname but never looks into __bases__ */
Obj    SetRecname_nobases ( Obj hdRec, Obj hdField, Obj hdVal ) {
    Obj  * ptRec;
    if ( GET_TYPE_BAG(hdRec) != T_REC )
        return Error("SetRecname: Record expected", 0, 0);
    ptRec = _FindRecnameRec_nobases(hdRec, hdField);
    if ( ptRec != 0 ) { /* field exists */
        ptRec[1] = hdVal;
        //  CHANGED_BAG(hdRec);
    } else { /* create a new field */
        Resize(hdRec, GET_SIZE_BAG(hdRec) + 2*SIZE_HD);
        SET_BAG(hdRec,  GET_SIZE_BAG(hdRec) / SIZE_HD - 2 ,  hdField );
        SET_BAG(hdRec,  GET_SIZE_BAG(hdRec) / SIZE_HD - 1 ,  hdVal );
    }
    return hdVal;
}
/* Converts string or an integer <hdNam> to T_RECNAM, if <hdNam> is T_RECNAM
   it is left alone. In case of other type, an error is signalled  */
Obj  RecnameObj ( Obj hdNam ) {
    UInt       k;              /* value from <rec>.(<int>)        */
    char                value [16];     /* <k> as a string                 */
    char                * p;            /* beginning of <k> in <value>     */

    if ( GET_TYPE_BAG(hdNam) != T_RECNAM ) {
        hdNam = EVAL(hdNam);
        if ( IsString( hdNam ) ) {
            return FindRecname( (char*)PTR_BAG(hdNam) );
        }
        else if ( GET_TYPE_BAG(hdNam) == T_RECNAM )
            return hdNam;
        else if ( GET_TYPE_BAG(hdNam) == T_INT && 0 <= HD_TO_INT(hdNam) ) {
            k = HD_TO_INT(hdNam);
            p = value + sizeof(value);  *--p = '\0';
            do { *--p = '0' + k % 10; } while ( (k /= 10) != 0 );
            return FindRecname( p );
        }
        else {
            return Error("<rec>.(<name>) <name> must be a string",0,0);
        }
    }
    else return hdNam;
}

/*
  SET_BAG(hdCall, 0,  ptRec[1] );
    SET_BAG(hdCall, 1,  hdL );
    SET_BAG(hdCall, 2,  hdR );
    hdResult = EVAL( hdCall );
    SET_BAG(hdCall, 0,  hdOperSym );
    SET_BAG(hdCall, 1,  0 );
    SET_BAG(hdCall, 2,  0 );
*/

Obj  FindOperatorRec(Obj hdRec, Obj hdOperField) {
    Obj  * ptRec;
    Obj  hdOpers, hdRealRec;

    /* if the right operand is a record look for the 'operations' element  */
    if ( GET_TYPE_BAG(hdRec) != T_REC ) return 0;
    ptRec = FindRecnameRec(hdRec, HdRnOp, &hdRealRec);
    if ( ptRec == 0 || GET_TYPE_BAG(ptRec[1]) != T_REC ) return 0;
    hdOpers = ptRec[1];

    /* if it was found and is a record look for the '+' element            */
    ptRec = FindRecnameRec(hdOpers, hdOperField, &hdRealRec);
    if ( ptRec == 0 ) return 0;

    return ptRec[1];
}

/* Evaluate an operator on two records by looking up an entry in .operations.<xxx> */
Obj  EvBinaryRecOperator ( Obj hdOperField, char *fldString, Obj hdL, Obj hdR,
                           Obj (*default_op)(Obj,Obj) ) {
    Bag           hdResult;
    Bag           hdOper;

    hdOper = FindOperatorRec(hdR, hdOperField);
    if ( hdOper == 0 ) {
        hdOper = FindOperatorRec(hdL, hdOperField);
        if ( hdOper == 0 ) {
            if ( default_op == 0 )
                return Error("Record: one operand must have '%s'",(Int)fldString,0);
            else
                return default_op(hdL, hdR);
        }
    }

    hdResult = NewBag(T_FUNCCALL, 3*SIZE_HD);

    SET_BAG(hdResult, 0,  hdOper );
    SET_BAG(hdResult, 1,  hdL );
    SET_BAG(hdResult, 2,  hdR );
    hdResult = EVAL(hdResult);

    return hdResult;
}

/* Evaluate an operator on one record by looking up an entry in .operations.<xxx> */
Obj  EvUnaryRecOperator ( Obj hdOperField, char *fldString, Obj hdRec,
                          Obj (*default_op)(Obj) ) {
    Bag           hdResult;
    Bag           hdOper;

    hdOper = FindOperatorRec(hdRec, hdOperField);
    if ( hdOper == 0 ) {
        if ( default_op == 0 )
            return Error("Record: operand must have '%g'",(Int)fldString,0);
        else
            return default_op(hdRec);
    }

    hdResult = NewBag(T_FUNCCALL, 2*SIZE_HD);

    SET_BAG(hdResult, 0,  hdOper );
    SET_BAG(hdResult, 1,  hdRec );
    hdResult = EVAL(hdResult);

    return hdResult;
}

/****************************************************************************
**
*F  EvRec( <hdRec> )  . . . . . . . . . . . . . . . . . . . evaluate a record
**
**  'EvRec' evaluates  the record <hdRec>.   Since records are constants  and
**  thus selfevaluating this simply returns <hdRec>.
*/
Bag       EvRec (Bag hdRec)
{
    return hdRec;
}


/****************************************************************************
**
*V  HdTilde . . . . . . . . . . . . . . . . . . . . . . . . . .  variable '~'
**
**  'HdTilde' is the handle of the variable bag of  the  variable  '~'.  This
**  variable can be used inside record and list literals to refer to the list
**  or record that is currently being created.  So for example '[ ~ ]' is the
**  list that contains itself as only element.
*/
Bag       HdTilde;


/****************************************************************************
**
*F  EvMakeRec(<hdLiteral>)  . . .  evaluate record literal to record constant
**
**  'EvMakeRec' evaluates the record literal, i.e., not yet evaluated, record
**  <hdLiteral> to a constant record.
**
**  'EvMakeRec' just calls 'MakeRec' telling it that the result goes into the
**  variable '~'.  Thus expressions in the variable  record can refer to this
**  variable and its subobjects to create objects that are not trees.
*/
Bag       EvMakeRec (Bag hdLiteral)
{
    Bag           hdRec;          /* handle of the result            */

    /* top level literal, make the list into '~'                           */
    if ( PTR_BAG(HdTilde)[0] == 0 ) {
        hdRec = MakeRec( HdTilde, 0, hdLiteral );
        SET_BAG(HdTilde, 0,  0 );
    }

    /* not top level, do not write the result somewhere                    */
    else {
        hdRec = MakeRec( 0, 0, hdLiteral );
    }

    /* return the result                                                   */
    return hdRec;
}

/****************************************************************************
**
*F  MakeRec(<hdDst>,<ind>,<hdLiteral>)  . evaluate record literal to constant
**
**  'MakeRec' evaluates the record literal <hdLiteral>  to a constant one and
**  puts  the result  into the bag  <hdDst>  at position <ind>.    <hdDst> is
**  either the variable '~', a list, or a record.
**
**  Because of literals like 'rec( a := rec( b := 1, c  := ~.a.b ) )' we must
**  enter the handle  of the result  into the superobject  before we begin to
**  evaluate the record literal.
**
**  A record  literal  is very much  like a  record, except that  at the even
**  places   the do not    yet contain  the  values  of  the  components, but
**  expressions, which yield the elements.   Evaluating a record literal thus
**  means looping over  the components, copying  the names at the odd entries
**  and evaluating the values at the even entries.
*/
Bag       MakeRec (Bag hdDst, UInt ind, Bag hdLiteral)
{
    Bag           hdRec;          /* handle of the result            */
    Bag           hdNam;          /* handle of component name bag    */
    Bag           hdVal;          /* handle of component value       */
    UInt       i;              /* loop variable                   */

    /* make the result bag and enter its handle in the superobject         */
    hdRec = NewBag( T_REC, GET_SIZE_BAG(hdLiteral) );
    if ( hdDst != 0 )  SET_BAG(hdDst, ind,  hdRec );

    /* loop over the components                                            */
    for ( i = 0; i < GET_SIZE_BAG(hdLiteral)/SIZE_HD/2; i++ ) {

        /* evaluate the name of the component if it is not constant        */
        hdNam = RecnameObj(PTR_BAG(hdLiteral)[2*i]);
        SET_BAG(hdRec, 2*i,  hdNam );

        /* evaluate and enter the value of this component                  */
        if ( GET_TYPE_BAG( PTR_BAG(hdLiteral)[2*i+1] ) == T_MAKELIST ) {
            MakeList( hdRec, 2*i+1, PTR_BAG(hdLiteral)[2*i+1] );
        }
        else if ( GET_TYPE_BAG( PTR_BAG(hdLiteral)[2*i+1] ) == T_MAKEREC ) {
            MakeRec( hdRec, 2*i+1, PTR_BAG(hdLiteral)[2*i+1] );
        }
        else {
            hdVal             = EVAL( PTR_BAG(hdLiteral)[2*i+1] );
            while ( hdVal == HdVoid )
                Error("Record: function must return a value",0,0);
            SET_BAG(hdRec, 2*i+1,  hdVal );
        }
    }

    /* return the record                                                   */
    return hdRec;
}


/****************************************************************************
**
*F  EvMakeTab(<hdLiteral>)  . . .  evaluate record literal to record constant
**
**  'EvMakeRec' evaluates the record literal, i.e., not yet evaluated, record
**  <hdLiteral> to a constant record.
**
**  'EvMakeRec' just calls 'MakeRec' telling it that the result goes into the
**  variable '~'.  Thus expressions in the variable  record can refer to this
**  variable and its subobjects to create objects that are not trees.
*/
Obj  MakeTab ( Obj hdLiteral );
Obj  EvMakeTab ( Obj hdLiteral ) {
    /* top level literal, make the list into '~'                           */
    if ( PTR_BAG(HdTilde)[0] == 0 ) {
        Obj hd = MakeTab(hdLiteral);
        SET_BAG(HdTilde, 0,  0 );
        return hd;
    }
    /* not top level, do not write the result somewhere                    */
    else return MakeTab(hdLiteral);
}

/****************************************************************************
**
*F  MakeTab(<hdLiteral>) . . . . . . . . . evaluate table literal to constant
**
**  'MakeRec' evaluates the record literal <hdLiteral>  to a constant one and
**  returns the result.
**
**  Because of literals like 'rec( a := rec( b := 1, c  := ~.a.b ) )' we must
**  enter the handle  of the result  into the superobject  before we begin to
**  evaluate the record literal.
**
**  A record  literal  is very much  like a  record, except that  at the even
**  places   the do not    yet contain  the  values  of  the  components, but
**  expressions, which yield the elements.   Evaluating a record literal thus
**  means looping over  the components, copying  the names at the odd entries
**  and evaluating the values at the even entries.
*/
Obj  MakeTab ( Obj hdLiteral ) {
    Obj   hdTab;          /* handle of the result            */
    Obj   hdNam;          /* handle of component name bag    */
    Obj   hdVal;          /* handle of component value       */
    UInt  i;              /* loop variable                   */

    /* make the result bag and enter its handle in the superobject         */
    hdTab = TableCreate( GET_SIZE_BAG(hdLiteral)/SIZE_HD * 3 / 2);

    /* loop over the components                                            */
    for ( i = 0; i < GET_SIZE_BAG(hdLiteral)/SIZE_HD/2; i++ ) {
        UInt pos;
        Obj hd;
        char * name;

        /* evaluate the name of the component if it is not constant        */
        hdNam = RecnameObj(PTR_BAG(hdLiteral)[2*i]);
        hdVal = EVAL( PTR_BAG(hdLiteral)[2*i+1] );
        if ( hdVal == HdVoid )
            Error("Table: function must return a value",0,0);

		name = RECNAM_NAME(hdNam);
        pos = TableLookup(hdTab, name, OFS_IDENT);
        hd = MakeIdent(name);
        SET_VAR_VALUE(hd, hdVal);
        TableAdd(hdTab, pos, hd);
    }

    /* return the record                                                   */
    return hdTab;
}

Obj  FunTabRec ( Obj hdCall ) {
    char * usage = "usage: TabRec( <rec> )";
    Obj hdRec;
    /* get and check the argument                                          */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  return Error(usage, 0,0);
    hdRec = EVAL( PTR_BAG(hdCall)[1] );
    if( GET_TYPE_BAG(hdRec) != T_REC ) return Error(usage,0,0);
    return MakeTab(hdRec);
}

/****************************************************************************
**
*F  EvRecElm( <hdElm> ) . . . . . . . . . . . . . . . .  get a record element
**
**  'EvRecElm' returns the value of 'PTR_BAG(<hdElm>)[0] . PTR_BAG(<hdElm>)[1]'.
**
**  '<record> . <name>'
**
**  This evaluates to the value of the record element with the name <name> in
**  the record <record>.  It is an  error  if  the  record  <record>  has  no
**  element with the name <name>.
**
**  'EvRecElm' simply iterates over  the record looking  for the record name.
**  If it is found the corresponding  value  is returned.  Note  that  it  is
**  not  neccessary  to compare  the strings of  the record names since every
**  record name is  store in a  unique record name  bag, i.e.,  no two record
**  name bags have the same string.  Thus we can simply compare handles.
*/
Bag     EvRecElm (Bag hdElm)
{
    Bag          hdRec = 0, hdNam = 0, hdValue = 0;
    Bag          * ptRec = 0;

    /* first get the record                                                */
    hdRec = EVAL( PTR_BAG(hdElm)[0] );
    if ( GET_TYPE_BAG(hdRec) != T_REC  && GET_TYPE_BAG(hdRec) != T_NAMESPACE )
        return Error("Record: %g, left operand must be a record", (Int)hdElm, 0);

    hdNam = RecnameObj( PTR_BAG(hdElm)[1] );

    /* then get the right operand, this is by construction a record name   */
    if ( GET_TYPE_BAG(hdRec) == T_REC ) {
        ptRec = FindRecnameRec(hdRec, hdNam, &hdValue);
        if ( ptRec == 0 )
            return Error("Record: element '%s' must have an assigned value",
                         (Int)PTR_BAG(hdNam), 0 );
        else hdValue = ptRec[1];
    }
    else {
        UInt pos = TableLookup(hdRec, (char*)PTR_BAG(hdNam), OFS_IDENT);
        Obj hd = PTR_BAG(hdRec)[pos];
        if ( hd == 0 )
            return Error("No identifier '%s' found", (Int)PTR_BAG(hdNam), 0);
        else if ( VAR_VALUE(hd) == 0 )
            return Error("Identifier '%s' has no value", (Int)PTR_BAG(hdNam), 0);
        else hdValue = VAR_VALUE(hd);
    }

    return hdValue;
}


/****************************************************************************
**
*F  EvRecAss( <hdAss> ) . . . . . . . . .  assign a value to a record element
**
**  'EvRecAss' assigns the value  'EVAL( <hdAss>[1] )'  to the  element  with
**  the name '<hdAss>[0][1]' in the record '<hdAss>[0][0]'.
**
**  '<record>.<name> := <expr>;'
**
**  This assigns the value  of the expression  <expr> to the element with the
**  name <name> in  the record <record>.  Further  references to this element
**  will  return this new  value, until another assignment is  performed.  If
**  the record has  no element  with the  name   <name> it is   automatically
**  extended.
*/
Bag       EvRecAss (Bag hdAss)
{
    Bag           hdRec,  hdNam,  hdVal;

    /* get the record                                                      */
    hdRec = EVAL( PTR_BAG(PTR_BAG(hdAss)[0])[0] );
    if ( GET_TYPE_BAG(hdRec) != T_REC  && GET_TYPE_BAG(hdRec) != T_NAMESPACE )
        return Error("Record Assignment: %g, left operand must be a record", (Int)hdAss, 0);

    hdNam = RecnameObj( PTR_BAG(PTR_BAG(hdAss)[0])[1] ); /* RHS */
    hdVal = EVAL( PTR_BAG(hdAss)[1] );               /* LHS */
    
    if ( hdVal == HdVoid )
        return Error("Record Assignment: function must return a value", 0, 0);

    if( GET_TYPE_BAG(hdRec) == T_REC ) {
        if ( GET_FLAG_BAG(hdRec, BF_PROTECT_VAR) )
            return Error("Record Assignment: %g, record is write-protected", (Int)hdAss, 0);
        return SetRecname_nobases(hdRec, hdNam, hdVal);
    }
    else {
        UInt pos = TableLookup(hdRec, (char*)PTR_BAG(hdNam), OFS_IDENT);
        Obj hd = PTR_BAG(hdRec)[pos];
        if ( hd == 0 )
            hd = TableAddIdent(hdRec, pos, (char*)PTR_BAG(hdNam));
        if ( GET_FLAG_BAG(hd, BF_PROTECT_VAR) ) {
            return Error("Assignment: variable %g is write-protected", (Int)hd, 0);
        }
        else
            return (SET_VAR_VALUE(hd, hdVal));
    }
}


/****************************************************************************
**
*F  SumRec( <hdL>, <hdR> )  . . . . . . . . . . . . . . . . sum of two record
*V  HdRnSum . . . . . . . . . . .  handle of the 'sum' record name bag, local
*V  HdCallSum . . . . . . . . . . . . . handle of the 'sum' function call bag
*V  HdStrSum
**
**  'SumRec' returns the sum of the two operands <hdL> and <hdR>, of which at
**  least one must be a record.
**
**  '<left> + <right>'
**
**  The sum of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '+', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '+' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> + <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnSum; /* hdOperField */
Bag       HdStrSum; /* hdOperSym */
Bag       HdCallSum; /* hdCall */

Obj  SumRec ( Obj hdL, Obj hdR ) {
    Obj ret = EvBinaryRecOperator(HdRnSum, "~.operations.+", hdL, hdR, 0);
    return ret;
}

/****************************************************************************
**
*F  DiffRec( <hdL>, <hdR> ) . . . . . . . . . . . .  difference of two record
*V  HdRnDiff  . . . . . . . . . . handle of the 'diff' record name bag, local
*V  HdCallDiff  . . . . . . . . . . .  handle of the 'diff' function call bag
*V  HdStrDiff
**
**  'DiffRec' returns the difference the of two operands  <hdL> and <hdR>, of
**  which at least one must be a record.
**
**  '<left> - <right>'
**
**  The difference of two records or an object and a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '-', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '-' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> - <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnDiff;
Bag       HdStrDiff;
Bag       HdCallDiff;

Obj  DiffRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnDiff, "~.operations.-", hdL, hdR, 0);
}


/****************************************************************************
**
*F  ProdRec( <hdL>, <hdR> ) . . . . . . . . . . . . . . product of two record
*V  HdRnProd  . . . . . . . . . . handle of the 'prod' record name bag, local
*V  HdCallProd  . . . . . . . . . . .  handle of the 'prod' function call bag
*V  HdStrProd
**
**  'ProdRec' returns the product of the two operands  <hdL>  and  <hdR>,  of
**  which at least one must be a record.
**
**  '<left> * <right>'
**
**  The product of two records or an  object  and  a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '*', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '*' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> * <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnProd;
Bag       HdStrProd;
Bag       HdCallProd;

Obj  ProdRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnProd, "~.operations.*", hdL, hdR, 0);
}

/****************************************************************************
**
*F  QuoRec( <hdL>, <hdR> )  . . . . . . . . . . . . .  quotient of two record
*V  HdRnQuo . . . . . . . . . . .  handle of the 'quo' record name bag, local
*V  HdCallQuo . . . . . . . . . . . . . handle of the 'quo' function call bag
*V  HdStrQuo
**
**  'QuoRec' returns the quotient of the two operands  <hdL>  and  <hdR>,  of
**  which at least one must be a record.
**
**  '<left> / <right>'
**
**  The quotient of two records or an object  and  a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '/', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '/' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> / <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnQuo;
Bag       HdStrQuo;
Bag       HdCallQuo;

Obj  QuoRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnQuo, "~.operations./", hdL, hdR, 0);
}

/****************************************************************************
**
*F  ModRec( <hdL>, <hdR> )  . . . . . . . . . . . . . remainder of two record
*V  HdRnMod . . . . . . . . . . .  handle of the 'mod' record name bag, local
*V  HdCallMod . . . . . . . . . . . . . handle of the 'mod' function call bag
*V  HdStrMod
**
**  'ModRec' returns the remainder of the two operands <hdL>  and  <hdR>,  of
**  which at least one must be a record.
**
**  '<left> mod <right>'
**
**  The remainder of two records or an object and  a  record  is  defined  as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'mod', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'mod' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> mod <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnMod;
Bag       HdStrMod;
Bag       HdCallMod;

Obj  ModRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnMod, "~.operations.mod", hdL, hdR, 0);
}


/****************************************************************************
**
*F  PowRec( <hdL>, <hdR> )  . . . . . . . . . . . . . . . power of two record
*V  HdRnPow . . . . . . . . . . .  handle of the 'pow' record name bag, local
*V  HdCallPow . . . . . . . . . . . . . handle of the 'pow' function call bag
*V  HdStrPow
**
**  'PowRec' returns the power of the two operands <hdL> and <hdR>, of  which
**  at least one must be a record.
**
**  '<left> ^ <right>'
**
**  The power of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '^', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '^' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> ^ <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnPow;
Bag       HdStrPow;
Bag       HdCallPow;

Obj  PowRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnPow, "~.operations.^", hdL, hdR, 0);
}

/****************************************************************************
**
*F  CommRec( <hdL>, <hdR> ) . . . . . . . . . . . .  commutator of two record
*V  HdRnComm  . . . . . . . . . . handle of the 'comm' record name bag, local
*V  HdCallComm  . . . . . . . . . . .  handle of the 'comm' function call bag
*V  HdStrComm
**
**  'CommRec' returns the commutator of the two operands <hdL> and  <hdR>, of
**  which at least one must be a record.
**
**  'Comm( <left>, <right> )'
**
**  The  commutator  of  two  records or an object and a record is defined as
**  follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'comm', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'comm' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> + <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnComm;
Bag       HdStrComm;
Bag       HdCallComm;

Obj  CommRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnComm, "~.operations.comm", hdL, hdR, 0);
}

/****************************************************************************
**
*F  EqRec( <hdL>, <hdR> ) . . . . . . . . . . .  test if two record are equal
*V  HdRnEq  . . . . . . . . . . . . handle of the 'eq' record name bag, local
*V  HdCallEq  . . . . . . . . . . . . .  handle of the 'eq' function call bag
*V  HdStrEq
**
**  'EqRec' returns 'HdTrue' two operands <hdL> and <hdR>, of which at  least
**  one must be a record, are equal and 'HdFalse' otherwise.
**
**  '<left> = <right>'
**
**  The equal of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '=', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '=' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> = <right>' is the value returned by this function.
**
**  In all other cases the operands are considered equal if both are  records
**  and they have the same names and all corresponding elements equal.
*/
Bag       HdRnEq;
Bag       HdStrEq;
Bag       HdCallEq;

Obj  DefaultEqRec ( Obj hdL, Obj hdR ) {
    UInt i, k;

    /* compare types and sizes */
    if ( GET_TYPE_BAG(hdL) != T_REC || GET_TYPE_BAG(hdR) != T_REC || GET_SIZE_BAG(hdL) != GET_SIZE_BAG(hdR) )
        return HdFalse;
    /* loop over all names of the left record                              */
    for ( i = 0; i < GET_SIZE_BAG(hdL)/(2*SIZE_HD); ++i ) {
		/* ignore __doc__ field */
		if (PTR_BAG(hdL)[2*i] != HdDocRecname) {
            /* look for that name in the right record                          */
            for ( k = 0; k < GET_SIZE_BAG(hdR)/(2*SIZE_HD); ++k ) {
                /* if found compare the elements                               */
                if ( PTR_BAG(hdL)[2*i] == PTR_BAG(hdR)[2*k] ) {
                    if ( PTR_BAG(hdL)[2*i+1] != PTR_BAG(hdR)[2*k+1]
                      && EQ( PTR_BAG(hdL)[2*i+1], PTR_BAG(hdR)[2*k+1] ) != HdTrue )
                        return HdFalse;
                    break;
                }
            }
            /* if not found the record are not equal                           */
            if ( k == GET_SIZE_BAG(hdR)/(2*SIZE_HD) )
                return HdFalse;
		}
    }
    /* everything matched, the records are equal                           */
    return HdTrue;
}

Obj  EqRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnEq, "~.operations.=", hdL, hdR, DefaultEqRec);
}

/****************************************************************************
**
*F  LtRec( <hdL>, <hdR> ) . . . . . . test if one record is less than another
*V  HdRnLt  . . . . . . . . . . . . handle of the 'lt' record name bag, local
*V  HdCallLt  . . . . . . . . . . . . .  handle of the 'lt' function call bag
*V  HdStrLt
**
**  'LtRec' returns 'HdTrue' the operand <hdL> is less than the operand <hdR>
**  and 'HdFalse' otherwise .  At least one of the operands must be a record.
**
**  '<left> < <right>'
**
**  The lt of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '<', which is a function,
**  or if the left operand <left> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element '<' which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> < <right>' is the value returned by this function.
**
**  In all other cases the operands are considered ltual if both are  records
**  and they have the same names and all corresponding elements ltual...
*/
Bag       HdRnLt;
Bag       HdStrLt;
Bag       HdCallLt;

Obj  DefaultLtRec ( Obj hdL, Obj hdR ) {
    UInt h, k;
    UInt i, j;
    Obj hdNam, hdVal;

    /* if no function is applied both operands must be records             */
    if ( GET_TYPE_BAG(hdL) < GET_TYPE_BAG(hdR) )
        return HdTrue;
    else if ( GET_TYPE_BAG(hdR) < GET_TYPE_BAG(hdL) )
        return HdFalse;

    /* sort the left record with a shellsort                               */
    h = 1;  while ( 9*h + 4 < GET_SIZE_BAG(hdL)/(2*SIZE_HD) )  h = 3*h + 1;
    while ( 0 < h ) {
        for ( i = h+1; i <= GET_SIZE_BAG(hdL)/(2*SIZE_HD); i++ ) {
            hdNam = PTR_BAG(hdL)[2*i-2];
            hdVal = PTR_BAG(hdL)[2*i-1];
            k = i;
            while ( h < k
                 && strcmp( RECNAM_NAME( hdNam ),
                              RECNAM_NAME( PTR_BAG(hdL)[2*(k-h)-2] ) ) < 0 ) {
                SET_BAG(hdL, 2*k-2,  PTR_BAG(hdL)[2*(k-h)-2] );
                SET_BAG(hdL, 2*k-1,  PTR_BAG(hdL)[2*(k-h)-1] );
                k -= h;
            }
            SET_BAG(hdL, 2*k-2,  hdNam );
            SET_BAG(hdL, 2*k-1,  hdVal );
        }
        h = h / 3;
    }

    /* sort the right record with a shellsort                              */
    h = 1;  while ( 9*h + 4 < GET_SIZE_BAG(hdR)/(2*SIZE_HD) )  h = 3*h + 1;
    while ( 0 < h ) {
        for ( i = h+1; i <= GET_SIZE_BAG(hdR)/(2*SIZE_HD); i++ ) {
            hdNam = PTR_BAG(hdR)[2*i-2];
            hdVal = PTR_BAG(hdR)[2*i-1];
            k = i;
            while ( h < k
                 && strcmp( RECNAM_NAME( hdNam ),
                              RECNAM_NAME( PTR_BAG(hdR)[2*(k-h)-2] ) ) < 0 ) {
                SET_BAG(hdR, 2*k-2,  PTR_BAG(hdR)[2*(k-h)-2] );
                SET_BAG(hdR, 2*k-1,  PTR_BAG(hdR)[2*(k-h)-1] );
                k -= h;
            }
            SET_BAG(hdR, 2*k-2,  hdNam );
            SET_BAG(hdR, 2*k-1,  hdVal );
        }
        h = h / 3;
    }

    /* now see what differs                                                */
	j = 1;
    for ( i = 1; i <= GET_SIZE_BAG(hdR)/(2*SIZE_HD); i++ ) {
		/* skip __doc__ on the right */
		if (PTR_BAG(hdR)[2*i-2] != HdDocRecname) {
			do {
			    if ( j > GET_SIZE_BAG(hdL)/(2*SIZE_HD) ) {
                    return HdTrue;
                }
			    /* skip __doc__ on the left */
			    if ( PTR_BAG(hdL)[2*j-2] != HdDocRecname ) break;
			    j++;
		    } while(1);

            if ( PTR_BAG(hdL)[2*j-2] != PTR_BAG(hdR)[2*i-2] ) {
                if ( strcmp( RECNAM_NAME( PTR_BAG(hdR)[2*i-2] ),
                           RECNAM_NAME( PTR_BAG(hdL)[2*j-2] ) ) < 0 ) {
                    return HdTrue;
                }
                else {
                    return HdFalse;
                }
            }
            else if ( EQ( PTR_BAG(hdL)[2*j-1], PTR_BAG(hdR)[2*i-1] ) == HdFalse ) {
                return LT( PTR_BAG(hdL)[2*j-1], PTR_BAG(hdR)[2*i-1] );
            }
			j++;
		}
    }

    /* the records are equal, or the right is a proper prefix of the left  */
    return HdFalse;
}

Obj  LtRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnLt, "~.operations.<", hdL, hdR, DefaultLtRec);
}


/****************************************************************************
**
*F  InRec( <hdL>, <hdR> ) . . . . . . . . . .  test if a record is in another
*V  HdRnIn  . . . . . . . . . . . . handle of the 'in' record name bag, local
*V  HdCallIn  . . . . . . . . . . . . .  handle of the 'in' function call bag
*V  HdStrIn
**
**  'InRec' returns 'HdTrue' the operand <hdL> is in the  operand  <hdR>  and
**  'HdFalse' otherwise .  At least the right operand must be a record.
**
**  '<left> in <right>'
**
**  The 'in' of two records or an object and a record is defined as follows:
**
**  If the right operand <right> is a record
**    and is has a element with the name 'operations', which is a record,
**    and this record has a element 'in', which is a function,
**  then this function is called with <left> and <right> as arguments
**    and '<left> in <right>' is the value returned by this function.
**
**  In all other cases an error is raised.
*/
Bag       HdRnIn;
Bag       HdStrIn;
Bag       HdCallIn;

/*N Yevgen Voronenko: original GAP semantics only allowed operations.in in
    right operand, here we change it to allow in both left and right operands */

Obj  InRec ( Obj hdL, Obj hdR ) {
    return EvBinaryRecOperator(HdRnIn, "~.operations.in", hdL, hdR, 0);
}


/****************************************************************************
**
*F  PrRec( <hdRec> )  . . . . . . . . . . . . . . . . . . . .  print a record
*V  HdRnPrint . . . . . . . . .  handle of the 'print' record name bag, local
*V  HdCallPrint . . . . . . . . . . . handle of the 'print' function call bag
*V  HdStrPrint
**
**  'PrRec' prints the record with the handle <hdRec>.
**
**  If <hdRec> has an element 'operations' which is a record
**    and this record as an element 'print' which is a function
**  then this function is called with <hdRec> as argument and should print it
**
**  In all other cases the record is printed in the following form:
**
**  'rec( <name> := <expr>,... )'
**
**  'PrRec' is also called to print variable records, i.e., records that have
**  not yet been evaluated.  They are always printed in the second form.
*/
Bag       HdRnPrint;
Bag       HdStrPrint;
Bag       HdCallPrint;

Obj DefaultPrRec ( Obj hdRec ) {
    UInt i;
    int  is_first_printed = 1;
    int  pr_populated = 0;
    /*N 05-Jun-90 martin 'PrRec' should be capable of ignoring elements    */
    /*N 05-Jun-90 martin 'PrRec' should support '~.<path>'                 */
    if (GET_TYPE_BAG(hdRec) == T_MAKETAB)
        Pr("%2>tab(",0,0);
    else
        Pr("%2>rec(",0,0);
    for ( i = 0; i < GET_SIZE_BAG(hdRec)/(2*SIZE_HD); ++i ) {
        /* print an ordinary record name                                   */
        if ( GET_TYPE_BAG( PTR_BAG(hdRec)[2*i] ) == T_RECNAM ) {
            char * name = RECNAM_NAME(PTR_BAG(hdRec)[2*i]);

            /* don't print fields that start with '_'. This allows to hide
               some information in records, when using methods with 'meth' */
            if(name != 0 && name[0]=='_') continue;

            if (! pr_populated++) Pr("\n%2>",0,0);
            if ( ! is_first_printed ) Pr("%2<,\n%2>",0,0);
            is_first_printed = 0;

            PrVarName(name);
        }
        /* print an evaluating record name                                 */
        else {
            if (! pr_populated++) Pr("\n%2>",0,0);
            Pr(" (",0,0);
            Print( PTR_BAG(hdRec)[2*i] );
            Pr(")",0,0);
        }
        /* print the component                                             */
        Pr("%< := %>",0,0);
        Print( PTR_BAG(hdRec)[2*i+1] );
    }
    if (pr_populated)
    Pr(" %4<)",0,0);
    else
        Pr("%2<)",0,0);
    return HdVoid;
}

void  PrRec ( Obj hdRec ) {
    EvUnaryRecOperator(HdRnPrint, "~.operations.Print", hdRec, DefaultPrRec);
}

/****************************************************************************
**
*F  PrRecElm( <hdElm> ) . . . . . . . . . . . . . . .  print a record element
**
**  'PrRecElm' prints the record element in the following form:
**
**  '<record> . <name>'
*/
void            PrRecElm (Bag hdElm)
{
    /* print the record                                                    */
    Pr( "%>", 0, 0 );
    Print( PTR_BAG(hdElm)[0] );
    /* print an ordinary record name                                       */
    if ( GET_TYPE_BAG( PTR_BAG(hdElm)[1] ) == T_RECNAM ) {
        char * name = RECNAM_NAME( PTR_BAG(hdElm)[1] );
        Pr("%<.%>",0,0);
        PrVarName(name);
        Pr("%<",0,0);
    }
    /* print an evaluating record name                                     */
    else {
        Pr( "%<.%>(", 0, 0 );
        Print( PTR_BAG(hdElm)[1] );
        Pr( ")%<", 0, 0 );
    }
}


/****************************************************************************
**
*F  PrRecAss( <hdAss> ) . . . . . . . . . . . . . . print a record assignment
**
**  'PrRecAss' prints the record assignment in the form:
**
**  '<record>.<name> := <expr>;'
*/
void            PrRecAss (Bag hdAss)
{
    Pr( "%2>", 0, 0 );
    Print( PTR_BAG(hdAss)[0] );
    Pr( "%< %>:= ", 0, 0 );
    Print( PTR_BAG(hdAss)[1] );
    Pr( "%2<", 0, 0 );
}


/****************************************************************************
**
*F  PrRecName( <hdName> ) . . . . . . . . . . . . . print a record field name
**
*/
void            PrRecName (Bag hdNam)
{
    Pr("RecName(\"", 0, 0);
    PrVarName(RECNAM_NAME(hdNam));
    Pr("\")", 0, 0);
}


/****************************************************************************
**
*F  FunRecName( <hdName> ) . . . . . . . . . . .  returns a record field name
**
*/
Bag      FunRecName (Bag hdCall)
{
    Bag           hdObjUneval, hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: RecName( <string> )",0,0);

    hdObjUneval = PTR_BAG(hdCall)[1];
    hdObj = EVAL(hdObjUneval);

    if ( hdObj == HdVoid )
        return Error("RecName: function must return a value",0,0);
    else if ( GET_TYPE_BAG(hdObj) != T_STRING )
        return Error("RecName: <%g> must be a string",
                     (Int)hdObjUneval, 0);
    else {
		char * st = (char*)PTR_BAG(hdObj);
        Bag hd = FindRecname(st);
        return hd;
    }
}

/****************************************************************************
**
*F  FunStringRecName( <hdName> ) . . .  convert record field name to a string
**
*/
Bag      FunStringRecName (Bag hdCall)
{
    Bag           hdObjUneval, hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: StringRecName( <recname> )",0,0);

    hdObjUneval = PTR_BAG(hdCall)[1];
    hdObj = EVAL(hdObjUneval);

    if ( hdObj == HdVoid )
        return Error("StringRecName: function must return a value",0,0);
    else if ( GET_TYPE_BAG(hdObj) != T_RECNAM )
        return Error("StringRecName: <%g> must be a record name",
                     (Int)hdObjUneval, 0);
    else
        return StringToHd( RECNAM_NAME(hdObj) );
}
/****************************************************************************
**
*F  FunIsRec( <hdCall> )  . . . . . . . . . . . . . internal function 'IsRec'
**
**  'IsRec'  returns 'true' if the object  <obj>  is  a  record  and  'false'
**  otherwise.  May cause an error if <obj> is an unbound variable.
*/
Bag       FunIsRec (Bag hdCall)
{
    Bag           hdObj;

    /* evaluate and check the argument                                     */
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )
        return Error("usage: IsRec( <obj> )",0,0);
    hdObj = EVAL( PTR_BAG(hdCall)[1] );
    if ( hdObj == HdVoid )
        return Error("IsRec: function must return a value",0,0);

    /* return 'true' if <obj> is a rational and 'false' otherwise          */
    if ( GET_TYPE_BAG(hdObj) == T_REC )
        return HdTrue;
    else
        return HdFalse;
}


/****************************************************************************
**
*F  FunRecFields( <hdCall> )  . . . . . . . . . . . . . .  list record fields
**
**  'FunRecFields' implements the internal function 'RecFields'.
**
**  'RecFields( <rec> )'
**
**  'RecFields' returns a list of strings representing all record  fields  of
**  the record <rec>.
**
**  You must use 'RecFields' if you want to make a selective copy of a record
**  for example to delete record fields arising from 'SetRecField'.
**
**  |    gap> r := rec();;
**      gap> i := 2^3-1;;
**      gap> s := ConcatenationString( "r", String(i) );;
**      gap> SetRecField( r, s, 0 );;
**      gap> r;
**      rec( r7 := 0 )
**      gap> RecFields( r );
**      [ "r7" ] |
*/
extern Bag HdDocRecname;

Bag       FunRecFields (Bag hdCall)
{
    Bag           hdRec,  hdNam;
    Bag           hdStr;
    Int                i, listpos;

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: RecFields( <rec> )",0,0);
    hdRec = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdRec) != T_REC )
        return Error("RecFields: <rec> must be a record",0,0);
    hdNam = NewBag( T_LIST, SIZE_PLEN_PLIST( GET_SIZE_BAG(hdRec) / SIZE_HD / 2 ) );
    SET_LEN_PLIST( hdNam, GET_SIZE_BAG(hdRec) / SIZE_HD / 2 );

    listpos = 1;
    /* list the record names                                               */
    for ( i = 1; i <= LEN_PLIST( hdNam ); i++ ) {

            hdStr = NewBag( T_STRING, GET_SIZE_BAG( PTR_BAG(hdRec)[2*i-2] ) );
            strncat( (char*)PTR_BAG(hdStr),
                       RECNAM_NAME( PTR_BAG(hdRec)[2*i-2] ),
                       strlen( RECNAM_NAME(PTR_BAG(hdRec)[2*i-2]) ) );
            SET_ELM_PLIST( hdNam, listpos++, hdStr );

    }
    SET_LEN_PLIST( hdNam, listpos-1 );

    /* return the list                                                     */
    return hdNam;
}

/****************************************************************************
**
*F  FunNumRecFields( <hdCall> )  . . . . . . . . number of fields in a record
**
**  'FunNumRecFields' implements the internal function 'NumRecFields', which
**  returns number of fields in a record.
*/
Bag       FunNumRecFields (Bag hdCall)
{
    Bag           hdRec;

    /* get and check the arguments                                         */
    if ( GET_SIZE_BAG(hdCall) != 2*SIZE_HD )
        return Error("usage: RecFields( <rec> )",0,0);
    hdRec = EVAL( PTR_BAG(hdCall)[1] );
    if ( GET_TYPE_BAG(hdRec) != T_REC )
        return Error("RecFields: <rec> must be a record",0,0);

    return INT_TO_HD( GET_SIZE_BAG(hdRec) / SIZE_HD / 2 );
}


/****************************************************************************
**
*F  InitRec() . . . . . . . . . . . . . . . .  intitialize the record package
**
**  'InitRec' initializes the record package.
*/
void            InitRec (void)
{
    Int                i;

    /* install the evaluation and printing functions                       */
    InstEvFunc( T_REC,     EvRec     );
    InstEvFunc( T_MAKEREC, EvMakeRec );
    InstEvFunc( T_MAKETAB, EvMakeTab );
    InstEvFunc( T_RECELM,  EvRecElm  );
    InstEvFunc( T_RECASS,  EvRecAss  );
    InstPrFunc( T_REC,     PrRec     );
    InstPrFunc( T_MAKEREC, PrRec     );
    InstPrFunc( T_RECELM,  PrRecElm  );
    InstPrFunc( T_RECASS,  PrRecAss  );
    InstPrFunc( T_RECNAM,  PrRecName );
    InstPrFunc( T_MAKETAB, PrRec     );

    /* install the magic variable '~'                                      */
    HdTilde = FindIdent( "~" );

    /* find the record name of the operations record                       */
    HdRnOp     = FindRecname( "operations" );

    /* find the record names of operators and create function call bags    */
    HdRnSum     = FindRecname( "+"     );
    HdCallSum   = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrSum    = FindIdent( "<rec1> + <rec2>" );
    SET_BAG(HdCallSum, 0,  HdStrSum );
    HdRnDiff    = FindRecname( "-"     );
    HdCallDiff  = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrDiff   = FindIdent( "<rec1> - <rec2>" );
    SET_BAG(HdCallDiff, 0,  HdStrDiff );
    HdRnProd    = FindRecname( "*"     );
    HdCallProd  = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrProd   = FindIdent( "<rec1> * <rec2>" );
    SET_BAG(HdCallProd, 0,  HdStrProd );
    HdRnQuo     = FindRecname( "/"     );
    HdCallQuo   = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrQuo    = FindIdent( "<rec1> / <rec2>" );
    SET_BAG(HdCallQuo, 0,  HdStrQuo );
    HdRnMod     = FindRecname( "mod"   );
    HdCallMod   = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrMod    = FindIdent( "<rec1> mod <rec2>" );
    SET_BAG(HdCallMod, 0,  HdStrMod );
    HdRnPow     = FindRecname( "^"     );
    HdCallPow   = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrPow    = FindIdent( "<rec1> ^ <rec2>" );
    SET_BAG(HdCallPow, 0,  HdStrPow );
    HdRnComm    = FindRecname( "Comm"  );
    HdCallComm  = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrComm   = FindIdent( "Comm( <rec1>, <rec2> )" );
    SET_BAG(HdCallComm, 0,  HdStrComm );
    HdRnEq      = FindRecname( "="     );
    HdCallEq    = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrEq     = FindIdent( "<rec1> = <rec2>" );
    SET_BAG(HdCallEq, 0,  HdStrEq );
    HdRnLt      = FindRecname( "<"     );
    HdCallLt    = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrLt     = FindIdent( "<rec1> < <rec2>" );
    SET_BAG(HdCallLt, 0,  HdStrLt );

    /* note that 'in' is special because it is not implemented as binop    */
    HdRnIn      = FindRecname( "in"    );
    HdCallIn    = NewBag( T_FUNCCALL, 3 * SIZE_HD );
    HdStrIn     = FindIdent( "<obj> in <rec>" );
    SET_BAG(HdCallIn, 0,  HdStrIn );

    /* note that 'print is special because it is a function not a binop    */
    HdRnPrint   = FindRecname( "Print" );
    HdCallPrint = NewBag( T_FUNCCALL, 2 * SIZE_HD );
    HdStrPrint  = FindIdent( "Print( <rec> )" );
    SET_BAG(HdCallPrint, 0,  HdStrPrint );

    HdBases = FindRecname("__bases__");

    for ( i = T_VOID; i < T_VAR; ++i ) {
        TabSum[  i ][ T_REC ] = SumRec;
        TabSum[  T_REC ][ i ] = SumRec;
        TabDiff[ i ][ T_REC ] = DiffRec;
        TabDiff[ T_REC ][ i ] = DiffRec;
        TabProd[ i ][ T_REC ] = ProdRec;
        TabProd[ T_REC ][ i ] = ProdRec;
        TabQuo[  i ][ T_REC ] = QuoRec;
        TabQuo[  T_REC ][ i ] = QuoRec;
        TabMod[  i ][ T_REC ] = ModRec;
        TabMod[  T_REC ][ i ] = ModRec;
        TabPow[  i ][ T_REC ] = PowRec;
        TabPow[  T_REC ][ i ] = PowRec;
        TabEq[   i ][ T_REC ] = EqRec;
        TabEq[   T_REC ][ i ] = EqRec;
        TabLt[   i ][ T_REC ] = LtRec;
        TabLt[   T_REC ][ i ] = LtRec;
        TabComm[ i ][ T_REC ] = CommRec;
        TabComm[ T_REC ][ i ] = CommRec;
    }

    /* install the internal functions                                      */
    InstIntFunc( "IsRec",         FunIsRec          );
    InstIntFunc( "RecFields",     FunRecFields      );
    InstIntFunc( "NumRecFields",  FunNumRecFields   );
    InstIntFunc( "RecName",       FunRecName        );
    InstIntFunc( "StringRecName", FunStringRecName  );
    InstIntFunc( "TabRec",        FunTabRec         );
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



