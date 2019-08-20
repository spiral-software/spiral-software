#ifndef IFACE_H
#define IFACE_H

#define IF_SUCCESS             1
#define IF_FAILURE             -1
#define EXEC_SUCCESS        1
#define EXEC_QUIT           2
#define EXEC_ERROR          4
#define INTER_EXIT          -1


#define ID_ORIGINAL         0 /* Original Interface must be entry 0 in the table */
#define ID_CONSOLE          1 /* Other entries can be placed anywhere */
#define ID_GUI              2
#define ID_TCPIP            3
#define ID_JAVARMI          4
#define ID_BATCH            5
#define IFACE_TABLE_SIZE    6 /* When you add a new entry to table, adjust this number */

#define DEFAULT_INTERFACE   ID_ORIGINAL

struct gap_iface
{ 
  int id; 
  char *name; 
  int (*main)(); 
  void (*write_callback)(char *, FILE *);
};

/*
 * execute the command stored in the input buffer
 * whatever the gap kernel outputs will be stored in to the
 * output buffer
 * return value will represent the condition of the kernel
 * SUCCESS / FAILURE / SIG_STOP / ERROR 
 */
int execute(char *input, char *output);

/*
 * Resets the internal data structure of the gap kernel
 * to initial state 
 * return value will be SUCCESS or FAILURE
 */
int reset();

/* This is a signal handler for the CTRL+C
 * This signal handler is used to catch user interrupt
 * and stops the execution that was sent by using the
 * execute function. execute function will return with
 * SIG_STOP immediately
 */
void *SIG_STOP_HANDLER();

#endif
