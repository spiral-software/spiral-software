
#include "buf_list.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int buf_list_init(buf_t *head)
{
  head->next = NULL;
  (head->buf)[0] = '\0';
  head->status = NEW_LIST;
  return BL_SUCCESS;
}

int buf_list_free(buf_t *head)
{
  char temp[512];
  
  while(buf_list_remove(head,temp))
    continue;
  
  return BL_SUCCESS;
}

int buf_list_insert(buf_t *head, char *buf)
{
  buf_t *curr;
  buf_t *new;

  curr = head;
  
  for(; curr->next != NULL; curr = curr->next);
  
  new = (buf_t *)malloc(sizeof(buf_t));
  strcpy(new->buf, buf);
  new->next = NULL;

  curr->next = new;

  return BL_SUCCESS;
}

int buf_list_remove(buf_t *head, char *buf)
{
  buf_t *temp;
  
  if(head->next == NULL)
    return BL_FAILURE;
  
  temp = head->next;
  head->next = head->next->next;
  
  strcpy(buf, temp->buf);
  free(temp);

  return BL_SUCCESS;
}
  
void print_buf_list(buf_t *head)
{
  buf_t *curr;
  
  if(head->next == NULL){
    printf("[Empty]\n");
  }
  else{
    curr = head;
    for(;curr->next != NULL; curr = curr->next){
      printf("[%s] -> ", curr->next->buf);
    }
    printf("[End]\n");
  }
}

/*
int main(int argc, char * agrv[]){
  buf_t *head;
  char temp[512];

  head = (buf_t *)malloc(sizeof(buf_t));
  
  buf_list_init(head);

  print_buf_list(head);
  
  buf_list_insert(head, "Test1");
  
  print_buf_list(head);

  buf_list_insert(head, "Test2");
  buf_list_insert(head, "Test3");

  print_buf_list(head);

  buf_list_remove(head, temp);
  
  print_buf_list(head);

  buf_list_remove(head,temp);
  
  buf_list_remove(head,temp);
  
  print_buf_list(head);
  printf("%s\n",temp);
}
*/
