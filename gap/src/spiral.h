#ifndef __SPIRAL_H__
#define __SPIRAL_H__

#include <stdio.h>
#include "conf.h"

/****************************************************************************
**
*F  InitLibName(argv, SyLibName, maxLen) . . . . . . . . .  sets library path
**
**  'InitLibName' initializes GAP library path variable by reading the  value
**  from libsys_conf - SPIRAL's configuration layer library.
**
**  The value is appended to SyLibname, so SyLibname must be already initial-
**  ized to a valid string. maxLen should  contain the  available storage  of
**  SyLibName.
*/
void            InitLibName ( char *progname, char *SyLibname , int maxLen );

/****************************************************************************
**
*F  InitSPIRAL() . . . . . . . . . . . .  initializes SPIRAL related packages
**
**  'InitSPIRAL' initializes packages needed by SPIRAL
*/
void            InitSPIRAL ( );

/****************************************************************************
**
*F  Props(<var>) . . . . . . returns property list associated with a variable
**
**  Each variable has a pointer to 'property list' which is a  special  place 
**  for  information  about the identifier. Property list  is  actually a GAP 
**  record, and  allows  arbitrary number  of  fields to be associated with a 
**  variable.
*/
Obj  Props ( Obj hd );

/****************************************************************************
**
*F  GetPromptString(char* RecFieldName) . . . . . . . returns prompt T_STRING
**
**  Reading value of GAPInfo.prompts.RecFieldName field. Returns bag that 
**  contains T_STRING if found or 0 if not found or field evaluated to 
**  another bag type. GAP uses this to read command line prompts. 
**  GAPInfo.prompts record may contain "spiral", "brk" and "dbg" fields.
*/

#define PROMPT_FIELD_SPIRAL "spiral"
#define PROMPT_FIELD_BRK    "brk"
#define PROMPT_FIELD_DBG    "dbg"

Bag GetPromptString(char* RecFieldName);

/****************************************************************************
**
*F  bag _ObjId(<bag>) . . . . . . . . . . . . . . . implements ObjId function
**
*/

Bag  _ObjId ( Bag hd );

#endif
