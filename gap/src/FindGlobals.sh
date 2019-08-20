echo '/* use FindGlobals.sh to generate most of this file, however '
echo '   there are portions outlined below that need to be added by hand */'
echo 
echo '#include "system.h"'
echo '#include "gasman.h"'
echo '#include <sys_conf/misc.h>'
echo
grep '*V' *.c | grep Hd | awk '{ print "extern Bag "$2";" }' | sort | uniq
echo 
echo 'void InitAllGlobals() {'
grep '*V' *.c | grep Hd | awk '{ print "    InitGlobalBag(&" $2 ", \""$2"\");" }' | sort | uniq
echo '}'
