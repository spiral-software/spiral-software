#include "stdio.h"
#include <stdlib.h>
#include "iface.h"

#include "iface_original.h"
#include "iface_console.h"
#include "iface_batch.h"
//#include "iface_gui.h"
//#include "iface_tcpip.h"
//#include "iface_rmi.h"

struct gap_iface gap_interface[IFACE_TABLE_SIZE] =
  {{ID_ORIGINAL, "original", original_main, original_printf}, // Working
   {ID_CONSOLE, "console", console_main, console_printf}, // Working 
   {ID_GUI, "gui", console_main, console_printf},  // Not implemented
   {ID_TCPIP, "tcpip", console_main, console_printf},
   //{ID_TCPIP, "tcpip", tcpip_main, tcpip_printf},  // kinda working 
   {ID_JAVARMI, "javarmi", console_main, console_printf},  // Not implemented
   {ID_BATCH, "batch", batch_main, batch_printf} // Working
};

