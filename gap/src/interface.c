/********************************************************************
 * interface.c
 *   
 * Joohoon Lee
 *  
 *
 * This file is used to send / recieve commands to the Spiral kernel
 * from outside programs such as the networking program for Spiral and
 * local GUI consoles 
 *
 */
#include "buf_list.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "iface.h"

#include        "system.h"              /* system dependent functions      */
#include        "memmgr.h"              /* dynamic storage manager         */
#include        "scanner.h"             /* reading of single tokens        */
#include        "eval.h"                /* evaluator main dispatcher       */
#include        "integer.h"             /* arbitrary size integers         */

#include        "idents.h"              /* 'InitIdents', 'FindIdent'       */
#include        "read.h"                /* 'ReadIt'                        */

#include        "list.h"                /* generic list package            */
#include        "plist.h"               /* 'LEN_PLIST', 'SET_LEN_PLIST',.. */
#include        "string.h"              /* 'IsString', 'PrintString'       */

#include        "statemen.h"            /* 'HdStat', 'StrStat'             */
#include        "function.h"            /* 'HdExec', 'ChangeEnv', 'PrintF' */
#include        "record.h"              /* 'HdCall*', 'HdTilde'            */

#include        "spiral.h"              /* InitSPIRAL, Try, Catch, exc     */
#include        "args.h"

#include        "interface.h"           /* New Spiral Interface            */
#include		"GapUtils.h"


extern struct gap_iface gap_interface[];

extern Bag            HdStack;
extern UInt        TopStack;

extern Bag       HdLast, HdLast2, HdLast3;
extern Bag       HdTime;


int interface_enable;


char static_input_buf[STATIC_INPUT_BUF_SIZE];
char static_output_buf[STATIC_OUTPUT_BUF_SIZE];



char *interface_write_input_nolist(char *line)
{
  strcpy(static_input_buf, line);
  return static_input_buf;
}

char *interface_read_input_nolist(char *line)
{
  strcpy(line, static_input_buf);
  return line;
}

char *interface_write_output_nolist(char *line)
{
  if(strlen(static_output_buf)+strlen(line) < STATIC_OUTPUT_BUF_SIZE)
    sprintf(static_output_buf, "%s%s", static_output_buf,line); 
  return static_output_buf;
}

char *interface_read_output_nolist(char *line)
{
  strcpy(line, static_output_buf);
  static_output_buf[0] = '\0';
  return line;
}


extern int original_main(void);

int start_interface(int interface_type)
{
  int ret = 1;
  
  while(ret != INTER_EXIT){
    ret = original_main();
  }
  
  return INTER_EXIT;
}
