#ifndef IFACE_H
#define IFACE_H

#define IF_SUCCESS             1
#define IF_FAILURE             -1
#define EXEC_SUCCESS        1
#define EXEC_QUIT           2
#define EXEC_ERROR          4
#define INTER_EXIT          -1


#define ID_ORIGINAL         0 /* Original Interface must be entry 0 in the table */

#define IFACE_TABLE_SIZE    1 /* When you add a new entry to table, adjust this number */

#define DEFAULT_INTERFACE   ID_ORIGINAL


/*
 * execute the command stored in the input buffer
 * whatever the gap kernel outputs will be stored in to the
 * output buffer
 * return value will represent the condition of the kernel
 * SUCCESS / FAILURE / SIG_STOP / ERROR 
 */
int execute(char *input, char *output);

#endif
