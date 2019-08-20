#include <stdio.h>
#include <stdlib.h>
#include "iface.h"

int original_main(void) {
  int exec_status, i;
  char input[4096];
  char output[4096];
  for(i=0; i<4096; ++i) input[i] = output[i] = 0;

  exec_status = 1;
  
  while(exec_status != EXEC_QUIT){
    exec_status = execute(input, output);
  }

  return INTER_EXIT;
}

void original_printf(char *data, FILE* fp){
  printf("%s", data);
}
