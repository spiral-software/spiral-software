/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef SMP2_H
#define SMP2_H

#if !defined(_WIN32) && !defined(_WIN64)
__inline long _InterlockedIncrement (long volatile *m);
#else
long  __cdecl _InterlockedIncrement(long volatile *Addend);
#endif

#if USE_SCHED_AFFINITY
#include <sched.h>
//the definition of sched_setaffinity inside sched.h is valid
//only for GCC, so here is the ICC def.
extern int sched_setaffinity (__pid_t __pid, size_t __cpusetsize,
                              unsigned long *__cpuset) __THROW;
#endif

#define INIT_BARRIER() {0, {1, 1}}

typedef struct { 
    volatile int id;
    volatile long val[2]; 
} barrier_t;

void barrier(int num_threads, int tid, barrier_t *b);
void set_affinity(int tid);

extern barrier_t GLOBAL_BARRIER;

#endif
