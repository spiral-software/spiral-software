#include <stdio.h>
#include <stdlib.h>
#include "iface.h"
#define BUF_SIZE 4096

int console_main(void) {
  int exec_status;
  char input[BUF_SIZE];
  char output[BUF_SIZE];

  exec_status = 1;
  
  while(exec_status != EXEC_QUIT){
    printf("inputs> ");
    fgets( input, 2048, stdin );
    exec_status = execute(input, output);
  }

  return INTER_EXIT;
}

void console_printf(char *data, FILE* fp){
  printf("console_callback:%s", data);
}
