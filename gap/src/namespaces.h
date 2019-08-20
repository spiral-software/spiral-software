void InitNamespaces( );

Obj PushNamespace ( Obj hdNS );
void PushNamespaces ( Obj hdNSList );
Obj PushGlobalNamespace ( Obj hdNS );
Obj PopNamespace ( );
int PopAllNamespaces ( );
Obj  NamespaceAdd ( Obj hdTo, Obj hdFrom );

Obj PushPackage ( Obj hdNS );
void PushPackages ( Obj hdNSList );
Obj PopPackage ( );

Obj StartPackage(char * name);
Obj GlobalPackage(char * name);
Obj StartPackageSpec(Obj spec);
Obj GlobalPackageSpec(Obj spec);
Obj EndPackage();

Obj StartPackage2( char * super, char * sub );
Obj GlobalPackage2( char * super, char * sub );

#define PGAP(nam) GlobalPackage2("gap", nam)
#define PEND      EndPackage
