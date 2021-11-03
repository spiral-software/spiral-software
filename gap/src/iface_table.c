#include "stdio.h"
#include <stdlib.h>
#include "iface.h"

#include "iface_original.h"

struct gap_iface gap_interface[IFACE_TABLE_SIZE] =
  {{ID_ORIGINAL, "original", original_main, original_printf}
};

