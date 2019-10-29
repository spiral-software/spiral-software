
#ifndef _SYSTEM_TYPES_H
#define _SYSTEM_TYPES_H


/****************************************************************************
**
*T  Char, Int1, Int2, Int4, Int, UChar, UInt1, UInt2, UInt4, UInt .  integers
**
**  'Char',  'Int1',  'Int2',  'Int4',  'Int',   'UChar',   'UInt1', 'UInt2',
**  'UInt4', 'UInt'  and possibly 'Int8' and 'UInt8' are the integer types.
**
**  Note that to get this to work, all files must be compiled with or without
**  '-DSYS_IS_64_BIT', not just "system.c".
**
**  '(U)Int<n>' should be exactly <n> bytes long
**  '(U)Int' should be the same length as a bag identifier
*/

/* 64 bit machines -- well alphas anyway                                   */
#ifdef SYS_IS_64_BIT
typedef char                    Char;
typedef char                    Int1;
typedef short int               Int2;
typedef int                     Int4;
#ifndef WIN64
typedef long int                Int8;
typedef long int                Int;
#else
typedef long long int			Int8;
typedef long long int			Int;
#endif
typedef unsigned char           UChar;
typedef unsigned char           UInt1;
typedef unsigned short int      UInt2;
typedef unsigned int            UInt4;
#ifndef WIN64
typedef unsigned long int       UInt8;
typedef unsigned long int       UInt;
#else
typedef unsigned long long int	UInt8;
typedef unsigned long long int	UInt;
#endif
#define	MAX_SMALL_INTEGER 0x0FFFFFFFFFFFFFFF
#define	MIN_SMALL_INTEGER (MAX_SMALL_INTEGER * -1)

/* 32bit machines                                                          */
#else
typedef char                    Char;
typedef char                    Int1;
typedef short int               Int2;
typedef long int                Int4;
typedef long long               Int8;
typedef long int                Int;
typedef unsigned char           UChar;
typedef unsigned char           UInt1;
typedef unsigned short int      UInt2;
typedef unsigned long int       UInt4;
typedef unsigned long long      UInt8;
typedef unsigned long int       UInt;

#define	MAX_SMALL_INTEGER 0x0FFFFFFF
#define	MIN_SMALL_INTEGER (MAX_SMALL_INTEGER * -1)
#endif

#define NUM_TO_INT(n)  ((Int) n)
#define NUM_TO_UINT(n)  ((UInt) n)

#endif // _SYSTEM_TYPES_H
