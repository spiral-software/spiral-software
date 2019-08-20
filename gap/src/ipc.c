#include <stdio.h>
#include <stdlib.h>
#include "iface.h"

int gap_ipc_main(void) {
  int exec_status;
  exec_status = 1;
  while(exec_status != EXEC_QUIT){
    exec_status = execute(NULL, NULL);
  }

  return INTER_EXIT;
}
