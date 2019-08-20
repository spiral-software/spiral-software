#include <stdio.h>
#include <stdlib.h>
#include "iface.h"

int batch_main(void) {
  return INTER_EXIT;
}

void batch_printf(char *data, FILE* fp){
  fprintf(fp, "%s", data);
}
