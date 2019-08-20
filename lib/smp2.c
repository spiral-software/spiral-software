#include "include/smp2.h"
#include <stdio.h>

barrier_t GLOBAL_BARRIER = INIT_BARRIER(); 

void set_affinity(int tid) {
#ifdef USE_WIN_THREADS
	HANDLE htid = GetCurrentThread();
	SetThreadAffinityMask(htid, 1<<tid);
//	SetThreadPriority(htid, THREAD_PRIORITY_HIGHEST);
#else
#ifdef USE_PTHREADS_AFFINITY
/*     	Beware that you need the NPTL-version of the pthreads 
	library for this to work (if you do not have it installed, 
	the compiler will complain that it does not know the 
	pthread_setaffinity_np function).

	you have to #include "nptl/pthread.h" not pthread.h, 
	which will probably use the linuxthreads header file which 
	doesn't have the affinity methods. you will also need to do a 
	-I/usr/include/nptl and -L/usr/lib/nptl. be aware you're breaking 
	your program's compatibility with linuxthreads as a result. 
*/
	unsigned long mask;
	mask = 1 << tid; 
	pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask); 

#elif USE_SCHED_AFFINITY
        unsigned long mask;
	mask = 1 << tid; 
        if (sched_setaffinity(0, sizeof(mask), &mask)< 0) {
          perror("sched_setaffinity");
        }
#else
	/* do nothing */
#endif
#endif
}

#if !defined(_WIN32) && !defined(_WIN64)
__inline  long _InterlockedIncrement(long volatile *m) {
  register int __res;
  __asm__ __volatile__ ("\n\
	movl	$1, %0\n\
	lock	xadd %0,(%1)\n\
	inc	%0\n\
	": "=a" (__res), "=q" (m): "1" (m));
  return __res;
}
#endif

void barrier(int num_threads, int tid, barrier_t *b) {
    int cur_id = b->id;
    //    printf("hello num_threads=%d, barrier=%d, tid=%d, val=%d\n", b->num_threads, b->id, tid, b->val[cur_id]);
    _InterlockedIncrement(&b->val[cur_id]);
    //    printf("    (upd) tid=%d, val=%d\n", tid, b->val[cur_id]);
    if (tid == 0) {
      while (b->val[cur_id] <= num_threads) /* val=1 means no threads touched it */
	    /* busy wait */ ;
      //	printf("    (get) tid=%d, val=%d\n", tid, b->val[cur_id]);
	b->id = (! b->id);
	b->val[b->id] = 1; /* set up new barrier, so threads don't fall thru */
	b->val[cur_id] = 0; /* current barrier -> fall thru */
	//	printf("    (set) tid=%d, val=%d\n", tid, b->val[cur_id]);
    }
    else while (b->val[cur_id] != 0);
    //    printf("    (get) tid=%d, val=%d\n", tid, b->val[cur_id]);
}
