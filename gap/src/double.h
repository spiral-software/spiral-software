/**********************************************************************
**
*F  DBL_INT(<hd>)  . . . . . . . convert small integer object to double
*F  DBL_OBJ(<hd>) . . . . . . . . . . . convert double object to double
*F  ObjDbl(<double>) . . . . . . . . create a double object from double
*F  DblAny(<hd>) . . . . . convert int/rational/string object to double
*/
#define DBL_INT(hd) ((double)HD_TO_INT(hd))
#define LONG_DBL_INT(hd) ((long double)HD_TO_INT(hd))
#define DBL_OBJ(hd) (*((double*)PTR_BAG(hd)))

Obj  ObjDbl(double d);
double DblString(char *st);
double DblAny(Obj hd);

long double LongDblAny(Obj hd);

void Init_Double();
