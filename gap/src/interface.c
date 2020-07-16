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
#include        "ipc.h"
#include		"GapUtils.h"

extern void InitGap (int argc, char** argv, int* stackBase);

extern int CURR_INTERFACE;
extern int LAST_INTERFACE;
extern struct gap_iface gap_interface[];

extern Bag            HdStack;
extern UInt        TopStack;

extern Bag       HdLast, HdLast2, HdLast3;
extern Bag       HdTime;


int input_buf;
int output_buf;

int interface_enable;
char interface_trigger_name[MAX_TRIGGER];

buf_t *input_buf_list[MAX_INPUTS];
buf_t *output_buf_list[MAX_OUTPUTS];

char static_input_buf[STATIC_INPUT_BUF_SIZE];
char static_output_buf[STATIC_OUTPUT_BUF_SIZE];

int reset_argc;
char **reset_argv;


/* You must call this function before using any of the fuction
   in the interface. Behavior when you call to other interface 
   function before initailization is undefined. The interface
   is not enabled when initialized. You must use use_interface
   function to actually enable the interface */

int interface_init()
{
  int i;
  for(i=0; i < MAX_INPUTS; i++){
    input_buf_list[i] = NULL;
    output_buf_list[i] = NULL;
  }

  interface_enable = NO_INTERFACE;
  interface_trigger_name[0] = '\0';
  input_buf = -1;
  output_buf = -1;
  return IF_SUCCESS;
}

/* You must provide a input buffer and output buffer to use.
   Those buffer descriptors have type of int, and must be created
   by call to create_input_buf(), and create_output_buf() function
   load_trig_name, unload_trig_name is used to control the use of
   interface inside the spiral local console */

int setup_interface(const char *trig_name, int input_buf_no, int output_buf_no)
{
  if(input_buf_list[input_buf_no] == NULL ||
     output_buf_list[output_buf_no] == NULL)
    return IF_FAILURE;
  
  interface_enable = NO_INTERFACE;
  strcpy(interface_trigger_name, trig_name);
  input_buf = input_buf_no;
  output_buf = output_buf_no;
  
  return IF_SUCCESS;
}


/* Determines what interface will be used at start up */
int InitInterface(int argc, char **argv)
{
  int i,j;
  LAST_INTERFACE = CURR_INTERFACE;
  CURR_INTERFACE = DEFAULT_INTERFACE;

  for(i=0; i<argc; i++){
    for(j=0; j<IFACE_TABLE_SIZE; j++){
      if(strcmp(argv[i], gap_interface[j].name) == 0)
	CURR_INTERFACE = gap_interface[j].id;
    }    
  }
  return IF_SUCCESS;
}

int enable_interface(const char *trig_name, int input_buf_no, int output_buf_no)
{
  if(input_buf_list[input_buf_no] == NULL ||
     output_buf_list[output_buf_no] == NULL)
    return IF_FAILURE;
  
  interface_enable = USE_INTERFACE;
  strcpy(interface_trigger_name, trig_name);
  input_buf = input_buf_no;
  output_buf = output_buf_no;
  
  return IF_SUCCESS;
}

int disable_interface()
{
  if(input_buf == -1 || output_buf == -1 ||
     interface_enable == NO_INTERFACE)
    return IF_FAILURE;
  
  interface_enable = NO_INTERFACE;
  interface_trigger_name[0] = '\0';
  input_buf = -1;
  output_buf = -1;
  return IF_SUCCESS;
}

/* Create a new input buffer and returns the descriptor 
   which is int type */
int create_input_buf()
{
  int i;
  for(i=0; (i < MAX_INPUTS && input_buf_list[i] != NULL); i++);
  
  if(i >= MAX_INPUTS){
    return IF_FAILURE;
  }
 
  input_buf_list[i] = (buf_t *)malloc(sizeof(buf_t));
  buf_list_init(input_buf_list[i]);
  
  return i;
}

/* Remove already created input buffer, whatever was stored in
   the buffer will get lost */
int remove_input_buf(int buf_no)
{
  if(input_buf_list[buf_no] == NULL)
    return IF_FAILURE;
  
  buf_list_free(input_buf_list[buf_no]);
  free(input_buf_list[buf_no]);
  input_buf_list[buf_no] = NULL;
  
  return IF_SUCCESS;
}  

/* Create a new input buffer and returns the descriptor 
   which is int type */
int create_output_buf()
{
  int i;
  for(i=0; (i < MAX_OUTPUTS && output_buf_list[i] != NULL); i++);
  
  if(i >= MAX_OUTPUTS){
    return IF_FAILURE;
  }
 
  output_buf_list[i] = (buf_t *)malloc(sizeof(buf_t));
  buf_list_init(output_buf_list[i]);
  
  return i;
}
	     
/* Remove already created output buffer, whatever was stored in
   the buffer will get lost */
int remove_output_buf(int buf_no)
{
  if(output_buf_list[buf_no] == NULL)
    return IF_FAILURE;
  
  buf_list_free(output_buf_list[buf_no]);
  free(output_buf_list[buf_no]);
  output_buf_list[buf_no] = NULL;
  
  return IF_SUCCESS;
}  

int check_load_trigger(const char *input)
{
  if(strlen(input) <= 5)
    return IF_FAILURE;
  if(strcmp(input+5, interface_trigger_name) != 0)
    return IF_FAILURE;

  return IF_SUCCESS;
}

int check_unload_trigger(const char *input)
{
  if(strlen(input) <= 7)
    return IF_FAILURE;
  if(strcmp(input+5, interface_trigger_name) != 0)
    return IF_FAILURE;
  
  return IF_SUCCESS;  
}

char *interface_read_input(char *line, int buf_no)
{
  buf_list_remove(input_buf_list[buf_no], line);
  return line;
}

char *interface_write_input(char *line, int buf_no)
{
  buf_list_insert(input_buf_list[buf_no], line);
  return line;
}

char *interface_read_output(char *line, int buf_no)
{
  buf_list_remove(output_buf_list[buf_no], line);
  return line;
}

char *interface_write_output(char *line, int buf_no)
{
  buf_list_insert(output_buf_list[buf_no], line);
  return line;
}


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

int interface_save_args(int argc, char *argv[])
{
  reset_argc = argc;
  reset_argv = argv;  
  return IF_SUCCESS;
}

int interface_reset(){  
  exc_type_t          e;
  Try {
    /* initialize everything                                             */
    InitGap( reset_argc, reset_argv, &reset_argc);
  }
  Catch(e) {
    exc_show();
    return IF_FAILURE;
  }
  
  return IF_SUCCESS;
}
int start_interface(int interface_type)
{
  int switch_flag = interface_type;
  
  while(switch_flag != INTER_EXIT){
    switch_flag = gap_interface[switch_flag].main();
    LAST_INTERFACE = CURR_INTERFACE;
    CURR_INTERFACE = switch_flag;
  }
  
  return INTER_EXIT;
}
