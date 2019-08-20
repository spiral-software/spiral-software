#include "system.h"
/* #include "types.h" */
/* #include        "flags.h" */			// defns & decls moved to memmgr.h
#include "memmgr.h"
#include "eval.h"
#include "integer.h"
#include "integer4.h"
#include "md5.h"
#include "args.h"
#include <sys/stat.h>
#include <string.h>
#ifdef WIN32
#include <windows.h>
#include <io.h>
#endif

/* does not return consistent results yet :(( */
UInt  _InternalHash ( Obj hd ) {
    if(!hd || GET_TYPE_BAG(hd)==T_NAMESPACE) return 0;
    else if(IS_INTOBJ(hd)) return HD_TO_INT(hd);
    else if(GET_FLAG_BAG(hd, BF_VISITED)) return 0;
    else if(GET_TYPE_BAG(hd)==T_VAR) return (Int) hd;   /* address of identifier */
    else if(GET_TYPE_BAG(hd)==T_VARAUTO) return (Int) EVAL(hd);

    {
	UInt result = 0;
	UInt i;
	UInt nhandles = NrHandles(GET_TYPE_BAG(hd), GET_SIZE_BAG(hd));
	/* don't mark variables, there is possibility of mark staying forever */
	if ( nhandles > 0 )
	  SET_FLAG_BAG(hd, BF_VISITED);    
	
	for (i = 0; i < nhandles; ++i )
	    result += _InternalHash(PTR_BAG(hd)[i]);

	for (i = nhandles * SIZE_HD; i < GET_SIZE_BAG(hd); ++i)
	    result += (UInt) (((unsigned char*)PTR_BAG(hd))[i]);

	return result;
    }
}

UInt  InternalHash ( Obj hd ) {
    UInt res = _InternalHash(hd);
    RecursiveClearFlag(hd, BF_VISITED);
    return res;
}

Obj  FunInternalHash ( Obj hdCall ) {
    char * usage = "usage: InternalHash( <obj> )";
    Obj hd;
    if ( GET_SIZE_BAG(hdCall) != 2 * SIZE_HD )  
		return Error(usage, 0,0);
    hd = EVAL(PTR_BAG(hdCall)[1]); 
    return INT_TO_HD(InternalHash(hd));
}

// MD5 GAP interface functions.

Bag FunMD5File(Bag hdString)
{
	Bag hdHash;

	// make sure we get one argument.
	if(GET_SIZE_BAG(hdString) != 2 * SIZE_HD)
		return Error("usage: MD5File( <filename> )", 0, 0);

	// eval the argument
    hdString = EVAL( PTR_BAG(hdString)[1] );
	
	// make sure it's a string
	if(GET_TYPE_BAG(hdString) != T_STRING)
		return Error("<filename> must be a string", 0, 0);

	// allocate a new bag to hold the hash (in T_INTPOS form)
	hdHash = NewBag(T_INTPOS, 8*sizeof(TypDigit));

	// call md5 function on file, result in hash.
	if(!MDFile(HD_TO_STRING(hdString), (char *) PTR_BAG(hdHash)))
		return Error("file not found", 0, 0);

	return hdHash;
}

Bag FunMD5String(Bag hdString)
{
	Bag hdHash;

	// make sure we get one argument.
	if(GET_SIZE_BAG(hdString) != 2 * SIZE_HD)
		return Error("usage: MD5String( <string> )", 0, 0);

	// eval the argument
    hdString = EVAL( PTR_BAG(hdString)[1] );
	
	// make sure the result is a string.
	if(GET_TYPE_BAG(hdString) != T_STRING)
		return Error("<string> is not a string", 0, 0);

	// allocate a new bag to hold the hash (in T_INTPOS form)
	hdHash = NewBag(T_INTPOS, 8*sizeof(TypDigit));

	// call md5 function on string, result in hash.
	MDString(HD_TO_STRING(hdString), (char *) PTR_BAG(hdHash));

	return hdHash;
}

Bag FunFileMTime(Bag hdString)
{
	Bag hdTime;
	unsigned long long mtime;

	// make sure we get one argument.
	if(GET_SIZE_BAG(hdString) != 2 * SIZE_HD)
		return Error("usage: FileMTime( <string> )", 0, 0);

	// eval the argument
    hdString = EVAL( PTR_BAG(hdString)[1] );
	
	// make sure the result is a string.
	if(GET_TYPE_BAG(hdString) != T_STRING)
		return Error("<string> is not a string", 0, 0);

	mtime = SyFmtime(
		HD_TO_STRING(hdString));

	// allocate a new bag to hold the hash (in T_INTPOS form)
	hdTime = NewBag(T_INTPOS, 4*sizeof(TypDigit));

	// copy time into our number.
	memcpy(PTR_BAG(hdTime), &mtime, sizeof(unsigned long long));

	return hdTime;
}

Bag FunGetPid(Bag hd)
{
	Int pid;
	Bag hdPid;

	pid = SyGetPid();
	hdPid = INT_TO_HD(pid);

	return hdPid;
}

/* 
   int SuperMakeDir(char *) 

   this function is capable of creating a multi-level directory
   in one call. if the dir already exists, it will return success.

   this function potentially mangles the string passed to it!!

   possible inputs include:
   "a/b/c", "a/b/c/", "/a/b", "a", "a/", "/a", "/a/"

   return 1 on success, 0 on failure!
*/

// this really should be in system.c.

Bag FunMakeDir(Bag hdString)
{
	char str[512];

	// make sure we get one argument.
	if(GET_SIZE_BAG(hdString) != 2 * SIZE_HD)
		return Error("usage: MakeDir( <dirname> )", 0, 0);

	// eval the argument
    hdString = EVAL( PTR_BAG(hdString)[1] );
	
	// make sure it's a string
	if(GET_TYPE_BAG(hdString) != T_STRING)
		return Error("<dirname> must be a string", 0, 0);
	
	// since we have to copy the string, make sure its not too long.
	if(strlen(HD_TO_STRING(hdString)) >= 512)
		return Error("<dirname> must be less than 511 chars long!", 0, 0);

	strcpy(str, HD_TO_STRING(hdString));

	if(!SuperMakeDir(str))
		return Error("could not create the directory!", 0, 0);

	return INT_TO_HD(1);
}

#ifndef WIN32
Bag FunWinGetValue(Bag hdkey)
{
	return HdFalse;
}
#else
Bag FunWinGetValue(Bag hdKey)
{
	char value[1024];
	int res;
	Bag hdString;

	// make sure we have one arg.
	if(GET_SIZE_BAG(hdKey) != 2 * SIZE_HD)
		return Error("usage: WinGetValue(<key>)", 0, 0);

	// eval the argument
	hdKey = EVAL(PTR_BAG(hdKey)[1]);

	// make sure it's a string
	if(GET_TYPE_BAG(hdKey) != T_STRING)
		return Error("<key> must be a string", 0, 0);

	res = WinGetValue(HD_TO_STRING(hdKey), value, 1024);

	// returned a integer.
	if(res < 0)
	{
		if(-res == 4)
			return INT_TO_HD(*((int *) value));
		else
			return Error("Unsupported integer size.", 0, 0);
	}
	else if(res > 0)
	{
		// the length returned by res includes the terminating null.
		hdString = NewBag(T_STRING, res);
		strncpy(HD_TO_STRING(hdString), value, res);

		return hdString;
	}

	return Error("Could not retrieve registry value.", 0, 0);
}
#endif

// insert double quote escaped spaces instead of spaces.
Bag FunWinPathFixSpaces(Bag hdString)
{
	char *str, *newstr;
	int n;

	// make sure we have one arg.
	if(GET_SIZE_BAG(hdString) != 2 * SIZE_HD)
		return Error("usage: WinPathFixSpaces(<path>)", 0, 0);

	// eval the argument
	hdString = EVAL(PTR_BAG(hdString)[1]);

	// make sure it's a string
	if(GET_TYPE_BAG(hdString) != T_STRING)
		return Error("<path> must be a string", 0, 0);


	// count the spaces
	str = HD_TO_STRING(hdString);
	for(n=0;(str = strchr(str, ' '));str++)
		n++;

	// subtract out already escaped spaces.
	str = HD_TO_STRING(hdString);
	for(;(str = strstr(str, "\" \""));str+=3)
		n--;

	// if we have at least one space
	if(n)
	{
		str = HD_TO_STRING(hdString);

		// alloc enough room for the string, trailing null, 
		// and the double quotes we will add.
		hdString = NewBag(T_STRING, strlen(str) + 1 + 2*n);

		newstr = HD_TO_STRING(hdString);

		// add double quotes around the spaces.
		for(;*str;str++)
		{

			// if we found a space
			if(*str == ' ')
			{
				*(newstr++) = '"';
				*(newstr++) = ' ';
				*(newstr++) = '"';
			}
			
			// need to skip already escaped spaces
			else if(str[0] == '"' && str[1] == ' ' && str[2] == '"')
			{
				*(newstr++) = *(str++);
				*(newstr++) = *(str++);
				*(newstr++) = *(str);
			}

			// copy everything else verbatim.
			else
				*(newstr++) = *str;
		}
		*newstr = 0; // terminating null.
	}

	return hdString;
}


Bag FunWinShortPathName(Bag origHdString)
{
#ifndef WIN32

	return HdFalse;

#else

	Bag newHdString;
	char *str, *dynstr, *newstr;
	int dynlen;

	// make sure we have one arg.
	if(GET_SIZE_BAG(origHdString) != 2 * SIZE_HD)
		return Error("usage: WinShortPathName(<path>)", 0, 0);

	// eval the argument
	origHdString = EVAL(PTR_BAG(origHdString)[1]);

	// make sure it's a string
	if(GET_TYPE_BAG(origHdString) != T_STRING)
		return Error("<path> must be a string", 0, 0);

	str = HD_TO_STRING(origHdString);
	dynlen = 2 * strlen(str);
	dynstr = (char *)malloc(dynlen);

	if (GetShortPathName(str, dynstr, dynlen) > 0)
	{
		newHdString = NewBag(T_STRING, strlen(dynstr) + 1);
		newstr = HD_TO_STRING(newHdString);
		strcpy(newstr, dynstr);
		free(dynstr);

		return newHdString;
	}
	else
	{
		free(dynstr);
		return origHdString;
	}

#endif /* WIN32 */
}
