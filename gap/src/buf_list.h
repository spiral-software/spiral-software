#ifndef _BUF_LIST_H_
#define _BUF_LIST_H_

#define BL_SUCCESS 1
#define BL_FAILURE 0
#define NEW_LIST  0x0
#define FINISH    0x1
#define ERROR     0x2
#define COMPUTING 0x4
#define PROGRESS  0x8


typedef struct buffer_list_struct{
	
  struct buffer_list_struct *next;
  char buf[512];
  int status;
  
}buf_t;

int buf_list_init(buf_t *head);
int buf_list_free(buf_t *head);
int buf_list_remove(buf_t *head, char *buf);
int buf_list_insert(buf_t *head, char *buf);
void print_buf_list(buf_t *head);

#endif /* _BUF_LIST_H_ */
