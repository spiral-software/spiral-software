/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */

#ifndef __THREADS_H__
#define __THREADS_H__

#if defined(_WIN32) || defined(_WIN64)
#define USE_WIN_THREADS
#else
#define USE_PTHREADS
//#define USE_PTHREADS_AFFINITY
#endif

#ifndef __cdecl
#define __cdecl
#endif

#ifdef USE_PTHREADS
#ifdef USE_PTHREADS_AFFINITY
#include <nptl/pthread.h>
#else
#include <pthread.h>
#endif
#define THREAD_FLD pthread_t pthread;
#define CREATE_THREAD(worker, data) \
	pthread_create(&(data.pthread), NULL, worker, &data); 
#endif

#ifdef USE_WIN_THREADS
#include <process.h>
#include <windows.h>
#define THREAD_FLD
#define CREATE_THREAD(worker, data)	_beginthread(worker, 0, &data)
#endif

#endif
