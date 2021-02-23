/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
#ifndef _RDTSC_H
#define _RDTSC_H

#define PROCESSOR_FREQ 750000000
#define COUNTER_LO(a) ((a).int32.lo)
#define COUNTER_HI(a) ((a).int32.hi)
#define COUNTER_VAL(a) ((a).int64)

#define COUNTER(a,b) \
	(((double)COUNTER_VAL(a))/(b))

#define COUNTER_DIFF(a,b,c) \
	(COUNTER(a,c)-COUNTER(b,c))

#define COUNTER_DIFF_SIMPLE(a,b) \
	(COUNTER_VAL(a)-COUNTER_VAL(b))

#define CYCLES		1
#define OPERATIONS	1
#define SEC		PROCESSOR_FREQ
#define MILI_SEC	(SEC/1E3)
#define MICRO_SEC	(SEC/1E6)
#define NANO_SEC	(SEC/1E9)

/* ==================== GNU C and possibly other UNIX compilers ===================== */
#if ! (defined(WIN32 ) || defined(WIN64))

#if defined(__GNUC__) || defined(__linux__)
#define VOLATILE __volatile__
#define ASM __asm__
#else
/* we can at least hope the following works, it probably won't */
#define ASM asm
#define VOLATILE 
#endif

#define myInt64 unsigned long long
#define INT32 unsigned int

typedef union
{       myInt64 int64;
        struct {INT32 lo, hi;} int32;
} tsc_counter;

#if defined(__ia64__)
	#if defined(__INTEL_COMPILER)
		#define RDTSC(tsc) (tsc).int64=__getReg(3116)
	#else
		#define RDTSC(tsc) ASM VOLATILE ("mov %0=ar.itc" : "=r" ((tsc).int64) )
	#endif

	#define CPUID() do{/*No need for serialization on Itanium*/}while(0)
#else
	#define RDTSC(cpu_c) \
		ASM VOLATILE ("rdtsc" : "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
	#define CPUID() \
		ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" )
#endif
	int rdtsc_works(void) {
		tsc_counter t0,t1;
		RDTSC(t0);
		RDTSC(t1);
		return COUNTER_DIFF(t1,t0,1) > 0;
	}
/* ======================== WIN 32/64 ======================= */
#else
	#define myInt64 signed __int64
	#define INT32 unsigned __int32

	typedef union
	{       myInt64 int64;
		struct {INT32 lo, hi;} int32;
	} tsc_counter;


#ifdef _MSC_VER

	#include <intrin.h>

	#define RDTSC(cpu_c)    { cpu_c.int64 = __rdtsc(); }
	#define CPUID()         { int CPUInfo[4]; __cpuid(CPUInfo, 0); }

#else /* not _MSC_VER */
      /* FIXME: add Win64 support */
	#define RDTSC(cpu_c)   \
	{       __asm rdtsc    \
			__asm mov (cpu_c).int32.lo,eax  \
			__asm mov (cpu_c).int32.hi,edx  \
	}

	#define CPUID() \
	{ \
		__asm mov eax, 0 \
		__asm cpuid \
	}

#endif
	int rdtsc_works(void) {
		tsc_counter t0,t1;
		__try {
		    RDTSC(t0);
		    RDTSC(t1);
		} __except ( 1) {
		    return 0;
		}
		return COUNTER_DIFF(t1,t0,1) > 0;
	}
#endif /* WIN 32/64 */

/*
#define RDTSC(cpu_c) \
{	asm("rdtsc"); 	\
	asm("mov %%eax, %0" : "=m" ((cpu_c).int32.lo) ); \
	asm("mov %%edx, %0" : "=m" ((cpu_c).int32.hi) ); \
}
*/

/*	Read Time Stamp Counter
	Read PMC 
#define RDPMC0(cpu_c) \
{		     	\
        __asm xor ecx,ecx	\
	__asm rdpmc	\
	__asm mov (cpu_c).int32.lo,eax	\
	__asm mov (cpu_c).int32.hi,edx	\
}
*/

// serialize reading the timer 
// #define TIME(a) CPUID(); RDTSC(a); CPUID();
#define TIME(a) RDTSC(a)


#endif // _RDTSC_H
