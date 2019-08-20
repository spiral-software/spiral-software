#ifndef _INTERFACE_H
#define _INTERFACE_H

#define MAX_INPUTS               16
#define MAX_OUTPUTS              16
#define MAX_TRIGGER              64
#define NO_INTERFACE              0
#define USE_INTERFACE             1
//#define IF_SUCCESS                   1
//#define FAILURE                   0
#define STATIC_INPUT_BUF_SIZE  4096
#define STATIC_OUTPUT_BUF_SIZE 4096

int InitInterface(int argc, char **argv);
int interface_init();
int setup_interface(const char *trig_name, int input_buf_no, int output_buf_no);
int enable_interface(const char *trig_name, int input_buf_no, int output_buf_no);
int disable_interface();
int create_input_buf();
int remove_input_buf(int buf_no);
int create_output_buf();
int remove_output_buf(int buf_no);
int check_load_trigger(const char *input);
int check_unload_trigger(const char *input);
char *interface_read_input(char *line, int buf_no);
char *interface_write_input(char *line, int buf_no);
char *interface_read_output(char *line, int buf_no);
char *interface_write_output(char *line, int buf_no);
char *interface_write_input_nolist(char *line);
char *interface_read_input_nolist(char *line);
char *interface_write_output_nolist(char *line);
char *interface_read_output_nolist(char *line);
int interface_save_args(int argc, char *argv[]);
int interface_reset();
int start_interface(int interface_type);

#endif // _INTERFACE_H

