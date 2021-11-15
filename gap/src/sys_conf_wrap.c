




/* Implementation : GAP */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "system.h"
#include "memmgr.h"
#include "integer.h"
#include        "objects.h"
#include		"string4.h"
#include "eval.h"
#include "idents.h"
#include "spiral.h"
#include "plist.h"
#include "args.h"
#include "double.h"
#include "GapUtils.h"






/* SWIG pointer structure */

typedef struct SwigPtrType {
  char               *name;               /* Datatype name                  */
  int                 len;                /* Length (used for optimization) */
  void               *(*cast)(void *);    /* Pointer casting function       */
  struct SwigPtrType *next;               /* Linked list pointer            */
} SwigPtrType;

/* Pointer cache structure */

typedef struct {
  int                 stat;               /* Status (valid) bit             */
  SwigPtrType        *tp;                 /* Pointer to type structure      */
  char                name[256];          /* Given datatype name            */
  char                mapped[256];        /* Equivalent name                */
} SwigCacheType;

/* Some variables  */

static int SwigPtrMax  = 64;           /* Max entries that can be currently held */
                                       /* This value may be adjusted dynamically */
static int SwigPtrN    = 0;            /* Current number of entries              */
static int SwigPtrSort = 0;            /* Status flag indicating sort            */
static int SwigStart[256];             /* Starting positions of types            */

/* Pointer table */
static SwigPtrType *SwigPtrTable = 0;  /* Table containing pointer equivalences  */

/* Cached values */

#define SWIG_CACHESIZE  8
#define SWIG_CACHEMASK  0x7
static SwigCacheType SwigCache[SWIG_CACHESIZE];  
static int SwigCacheIndex = 0;
static int SwigLastCache = 0;

/*
 * Keep essential bits, rest is for the scrapheap
 */

/* Make a pointer value string */


void SWIG_MakePtr(char *_c, const void *_ptr, char *type) {
  static char _hex[16] =
  {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
   'a', 'b', 'c', 'd', 'e', 'f'};
  UInt _p, _s;
  char _result[20], *_r;    /* Note : a 64-bit hex number = 16 digits */
  _r = _result;
  _p = (UInt) _ptr;
  if (_p > 0) {
    while (_p > 0) {
      _s = _p & 0xf;
      *(_r++) = _hex[_s];
      _p = _p >> 4;
    }
    *_r = '_';
    while (_r >= _result)
      *(_c++) = *(_r--);
  } else {
    strcpy (_c, "NULL");
  }
  if (_ptr)
    strcpy (_c, type);
}

/* Sort comparison function */
static int swigsort(const void *data1, const void *data2) {
	SwigPtrType *d1 = (SwigPtrType *) data1;
	SwigPtrType *d2 = (SwigPtrType *) data2;
	return strcmp(d1->name,d2->name);
}

/* Binary Search function */
static int swigcmp(const void *key, const void *data) {
  char *k = (char *) key;
  SwigPtrType *d = (SwigPtrType *) data;
  return strncmp(k,d->name,d->len);
}


/* Function for getting a pointer value */

char *SWIG_GetPtr(char *_c, void **ptr, char *_t)
{
  UInt _p;
  char temp_type[256];
  char *name;
  int  i, len;
  SwigPtrType *sp,*tp;
  SwigCacheType *cache;
  int  start, end;
  _p = 0;

  /* Pointer values must start with leading underscore */
  if (*_c == '_') {
      _c++;
      /* Extract hex value from pointer */
      while (*_c) {
	  if ((*_c >= '0') && (*_c <= '9'))
	    _p = (_p << 4) + (*_c - '0');
	  else if ((*_c >= 'a') && (*_c <= 'f'))
	    _p = (_p << 4) + ((*_c - 'a') + 10);
	  else
	    break;
	  _c++;
      }

      if (_t) {
	if (strcmp(_t,_c)) { 
	  if (!SwigPtrSort) {
	    qsort((void *) SwigPtrTable, SwigPtrN, sizeof(SwigPtrType), swigsort); 
	    for (i = 0; i < 256; i++) {
	      SwigStart[i] = SwigPtrN;
	    }
	    for (i = SwigPtrN-1; i >= 0; i--) {
	      SwigStart[(int) (SwigPtrTable[i].name[1])] = i;
	    }
	    for (i = 255; i >= 1; i--) {
	      if (SwigStart[i-1] > SwigStart[i])
		SwigStart[i-1] = SwigStart[i];
	    }
	    SwigPtrSort = 1;
	    for (i = 0; i < SWIG_CACHESIZE; i++)  
	      SwigCache[i].stat = 0;
	  }
	  
	  /* First check cache for matches.  Uses last cache value as starting point */
	  cache = &SwigCache[SwigLastCache];
	  for (i = 0; i < SWIG_CACHESIZE; i++) {
	    if (cache->stat) {
	      if (strcmp(_t,cache->name) == 0) {
		if (strcmp(_c,cache->mapped) == 0) {
		  cache->stat++;
		  *ptr = (void *) _p;
		  if (cache->tp->cast) *ptr = (*(cache->tp->cast))(*ptr);
		  return (char *) 0;
		}
	      }
	    }
	    SwigLastCache = (SwigLastCache+1) & SWIG_CACHEMASK;
	    if (!SwigLastCache) cache = SwigCache;
	    else cache++;
	  }
	  /* We have a type mismatch.  Will have to look through our type
	     mapping table to figure out whether or not we can accept this datatype */

	  start = SwigStart[(int) _t[1]];
	  end = SwigStart[(int) _t[1]+1];
	  sp = &SwigPtrTable[start];
	  while (start < end) {
	    if (swigcmp(_t,sp) == 0) break;
	    sp++;
	    start++;
	  }
	  if (start >= end) sp = 0;
	  /* Try to find a match for this */
	  if (sp) {
	    while (swigcmp(_t,sp) == 0) {
	      name = sp->name;
	      len = sp->len;
	      tp = sp->next;
	      /* Try to find entry for our given datatype */
	      while(tp) {
		if (tp->len >= 255) {
		  return _c;
		}
		strcpy(temp_type,tp->name);
		strncat(temp_type,_t+len,255-tp->len);
		if (strcmp(_c,temp_type) == 0) {
		  
		  strcpy(SwigCache[SwigCacheIndex].mapped,_c);
		  strcpy(SwigCache[SwigCacheIndex].name,_t);
		  SwigCache[SwigCacheIndex].stat = 1;
		  SwigCache[SwigCacheIndex].tp = tp;
		  SwigCacheIndex = SwigCacheIndex & SWIG_CACHEMASK;
		  
		  /* Get pointer value */
		  *ptr = (void *) _p;
		  if (tp->cast) *ptr = (*(tp->cast))(*ptr);
		  return (char *) 0;
		}
		tp = tp->next;
	      }
	      sp++;
	      /* Hmmm. Didn't find it this time */
	    }
	  }
	  /* Didn't find any sort of match for this data.  
	     Get the pointer value and return the received type */
	  *ptr = (void *) _p;
	  return _c;
	} else {
	  /* Found a match on the first try.  Return pointer value */
	  *ptr = (void *) _p;
	  return (char *) 0;
	}
      } else {
	/* No type specified.  Good luck */
	*ptr = (void *) _p;
	return (char *) 0;
      }
  } else {
    if (strcmp (_c, "NULL") == 0) {
	*ptr = (void *) 0;
	return (char *) 0;
    }
    *ptr = (void *) 0;	
    return _c;
  }
}


static char * _usage_config_demand_val = "config_demand_val (char *key_name)";
Obj _wrap_config_demand_val(Obj argv) {
    char * usage = _usage_config_demand_val;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    config_val_t * _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<key_name> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (config_val_t *)config_demand_val(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
	result = NewBag(T_STRING, 20+strlen("_config_val_t_p"));
	SWIG_MakePtr((char*) PTR_BAG(result), (void *) _result, "_config_val_t_p");
    }
    return result;
}

static char * _usage_config_get_val = "config_get_val (char *key_name)";
Obj _wrap_config_get_val(Obj argv) {
    char * usage = _usage_config_get_val;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    config_val_t * _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<key_name> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (config_val_t *)config_get_val(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
	result = NewBag(T_STRING, 20+strlen("_config_val_t_p"));
	SWIG_MakePtr((char*) PTR_BAG(result), (void *) _result, "_config_val_t_p");
    }
    return result;
}

#define config_val_strval_get(_swigobj) ((char *) _swigobj->strval)
static char * _usage_config_val_t_strval_get = "config_val_t_strval_get (config_val_t *)";
Obj _wrap_config_val_t_strval_get(Obj argv) {
    char * usage = _usage_config_val_t_strval_get;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    char * _result;
    config_val_t * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {
	char * ptr_string = HdToString(ELM_ARGLIST(argv, 1), 
		         "<arg 0> must be a _config_val_t_p pointer.\nUsage: %s", (Int)usage, 0); 
	if( SWIG_GetPtr(ptr_string,(void **) &_arg0, "_config_val_t_p") )
	    /* type error :  _config_val_t_p expected" */
	    return Error("<arg 0> must be a _config_val_t_p pointer.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (char *)config_val_strval_get(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = StringToHd(_result); 
    }
    return result;
}




/*== Helper Functions to convert sys_conf datatypes to GAP ==*/

Bag gap_config_val_t(config_val_t * val) {
	Bag hd = 0;
	if(val==0) 
	    return HdFalse;
	switch(val->type) {
	case VAL_INT: return INT_TO_HD(val->intval);
	case VAL_STR: C_NEW_STRING(hd, val->strval); return hd;
	case VAL_FLOAT: return ObjDbl((double)val->floatval); 
	default: ASSERT("val->type is invalid" == 0); 
	    return 0; /* never reached */
	}
}

static char * _usage_sys_exists = "sys_exists (const char *fname)";
Obj _wrap_sys_exists(Obj argv) {
    char * usage = _usage_sys_exists;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    int  _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<fname> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (int )sys_exists(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = IntToHd(_result); 
    }
    return result;
}


static char * _usage_sys_mkdir = "sys_mkdir (const char *name)";
Obj _wrap_sys_mkdir(Obj argv) {
    char * usage = _usage_sys_mkdir;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    int  _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<name> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (int )sys_mkdir(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = IntToHd(_result); 
    }
    return result;
}

static char * _usage_sys_rm = "sys_rm (const char *name)";
Obj _wrap_sys_rm(Obj argv) {
    char * usage = _usage_sys_rm;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    int  _result;
    char * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {	
	_arg0 = (char *) HdToString(ELM_ARGLIST(argv, 1), 
	   "<name> must be a String.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (int )sys_rm(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
        result = IntToHd(_result); 
    }
    return result;
}

static char * _usage_gap_config_val_t = "gap_config_val_t (config_val_t *val)";
Obj _wrap_gap_config_val_t(Obj argv) {
    char * usage = _usage_gap_config_val_t;
    int  argc = GET_SIZE_BAG(argv) / SIZE_HD;
    Obj  result;

    Bag  _result;
    config_val_t * _arg0;

    /* Check number of arguments */ 
    if ((argc < 2) || (argc > 2)) 
        return Error("Wrong number of arguments.\nUsage: %s", (Int)usage, 0);
    /* Get argument0 */ 
    {
	char * ptr_string = HdToString(ELM_ARGLIST(argv, 1), 
		         "<val> must be a _config_val_t_p pointer.\nUsage: %s", (Int)usage, 0); 
	if( SWIG_GetPtr(ptr_string,(void **) &_arg0, "_config_val_t_p") )
	    /* type error :  _config_val_t_p expected" */
	    return Error("<val> must be a _config_val_t_p pointer.\nUsage: %s", (Int)usage, 0); 
    }

    /*===== Call the C function =====*/ 
    _result = (Bag )gap_config_val_t(_arg0);
    /*===============================*/ 

    /* Convert the result */ 
    {
	result = (Bag) _result;
    }
    return result;
}


void Init_sys_conf(void) {

    InstIntFunc("config_get_val", _wrap_config_get_val);
    InstIntFunc("config_demand_val", _wrap_config_demand_val);
    InstIntFunc("sys_exists", _wrap_sys_exists);
    InstIntFunc("sys_mkdir", _wrap_sys_mkdir);
    InstIntFunc("sys_rm", _wrap_sys_rm);

    InstIntFunc("config_val_t_strval_get", _wrap_config_val_t_strval_get);

    InstIntFunc("gap_config_val_t", _wrap_gap_config_val_t);
	
    
}

