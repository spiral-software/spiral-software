
#ifndef __BIN_NS_H__
#define __BIN_NS_H__

extern Bag FindInPrivatePackages(char * name, int mode_rd);

extern Obj PushPrivatePackage ( Obj hdNS );
extern Obj PopPrivatePackage (void);

#endif
