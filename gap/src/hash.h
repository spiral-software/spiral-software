#ifndef _HASH_H
#define _HASH_H

extern Obj  FunInternalHash ( Obj hdCall );
extern Bag FunMD5File(Bag hdString);
extern Bag FunGetPid(Bag hd);
extern Bag FunFileMTime(Bag hdString);
extern Bag FunMakeDir(Bag hdString);
extern Bag FunMD5String(Bag hdString);
extern Bag FunWinGetValue(Bag length);
extern Bag FunWinPathFixSpaces(Bag hdString);
extern Bag FunWinShortPathName(Bag hdString);

#endif // _HASH_H
