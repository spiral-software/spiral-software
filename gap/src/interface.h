#ifndef _INTERFACE_H
#define _INTERFACE_H

#define MAX_INPUTS               16
#define MAX_OUTPUTS              16

#define STATIC_INPUT_BUF_SIZE  4096
#define STATIC_OUTPUT_BUF_SIZE 4096



char *interface_write_input_nolist(char *line);
char *interface_read_input_nolist(char *line);
char *interface_write_output_nolist(char *line);
char *interface_read_output_nolist(char *line);

int start_interface(int interface_type);

#endif // _INTERFACE_H

