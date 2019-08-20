/* Fast Fourier Transform (.c/.cpp-file)
   =====================================

   Sebastian Egner,
   in GNU-C v2.7.2, Visual-C++ v4.2, or Borland C++ v5.01
   The module is both a C and a C++ source.

   history
     18.06.96 q-Power twiddles are being enumerated
     19.06.96 Rader, Cooley-Tukey, Bluestein
     20.06.96 fft_new_<method> takes smaller (fft_t *)fft's
     15.08.96 reimplementation
     21.08.96 cleaned; apply_perm reimplemented
     22.08.96 generic arithmetics introduced
     23.08.96 small2,3,4 reimplemented
     24.08.96 small5,6,7,8,9 implemented
     28.08.96 printer/parser for FFT-trees
     22.11.96 improving portability (fewer warnings)
     23.11.96 buffers with alloca() (=> no buffers in fft-obj.)
     25.11.96 ported to Borland-C++ and MS-C++ again;
              rootsOfUnity-recurrence with (double)
     29.03.97 Bluestein-optimization
      1.04.97 included to DigiOpt again
      3.04.98 bug fixed: delete_small() was missing
     11.08.00 bug fixed (Gavin Haentjens, CMU): small malloc in fft_parse
     18.06.02 bug fixed: Apply_perm macro used N instead of (N)
     28.08.02 malloc -> xmalloc from spiral
	 25.06.19 Use malloc not xmalloc(); free not xfree(), decouple from libsys_conf

   compile (GNU-C): gcc -O2 -Wall -o fftest fftest.c fft.c -lm
   compile (HP-cc): cc -Aa -O -o fftest fftest.c fft.c -lm
   compile (BCC):   bcc32 -O2 fftest fft
   compile (VisC):  <project-workspace: MaximizeSpeed, Warn.-Level 3
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


#include "fft.h"

/* Exceptions where Bluestein is to be used (from profiling)
   =========================================================
   
   Use 'fftprof.g' (a GAP-program) and 'fftest' (an executable)
   to profile the actual architecture and implementation.

   Then edit 'tmp1' by replacing '[' -> '{' and ']' -> '}'
   and insert the file at this point.
*/

static const int fft_bluesteinExceptionTable[][2] = { 

  /* "Linux-PC 486DX (66Mhz, 16MB); total Bluestein-improved" */

  { 47, 96 }, { 94, 192 }, { 107, 224 }, { 139, 280 }, { 141, 288 }, 
  { 167, 360 }, { 179, 360 }, { 188, 384 }, { 214, 448 }, { 235, 480 }, 
  { 263, 576 }, { 269, 576 }, { 277, 576 }, { 278, 576 }, { 282, 576 }, 
  { 283, 576 }, { 317, 640 }, { 321, 672 }, { 329, 672 }, { 334, 672 }, 
  { 347, 720 }, { 358, 720 }, { 359, 720 }, { 376, 768 }, { 383, 768 }, 
  { 417, 840 }, { 419, 840 }, { 423, 896 }, { 428, 896 }, { 461, 960 }, 
  { 467, 960 }, { 470, 960 }, { 479, 960 }, { 499, 1024 }, { 501, 1024 }, 
  { 503, 1024 }, { 517, 1080 }, { 526, 1080 }, { 529, 1120 }, { 535, 1080 }, 
  { 537, 1080 }, { 538, 1080 }, { 554, 1120 }, { 556, 1120 }, { 557, 1120 }, 
  { 564, 1152 }, { 566, 1152 }, { 587, 1260 }, { 599, 1260 }, { 611, 1260 }, 
  { 619, 1260 }, { 634, 1280 }, { 642, 1344 }, { 643, 1344 }, { 658, 1344 }, 
  { 659, 1344 }, { 668, 1344 }, { 691, 1440 }, { 694, 1440 }, { 695, 1440 }, 
  { 705, 1440 }, { 709, 1440 }, { 716, 1440 }, { 718, 1440 }, { 719, 1440 }, 
  { 743, 1536 }, { 749, 1536 }, { 766, 1536 }, { 787, 1680 }, { 789, 1680 }, 
  { 797, 1680 }, { 799, 1680 }, { 807, 1680 }, { 823, 1680 }, { 827, 1680 }, 
  { 829, 1680 }, { 831, 1680 }, { 834, 1680 }, { 835, 1680 }, { 839, 1680 }, 
  { 849, 1792 }, { 857, 1792 }, { 863, 1792 }, { 887, 1792 }, { 893, 1792 }, 
  { 895, 1792 }, { 922, 2016 }, { 934, 1920 }, { 940, 1920 }, { 941, 1920 }, 
  { 951, 1920 }, { 958, 1920 }, { 963, 2016 }, { 967, 2016 }, { 973, 2048 }, 
  { 987, 2048 }, { 997, 2048 }, { 998, 2016 }, { 1002, 2048 }, 
  { 1006, 2048 }, { 1013, 2048 }, { 1019, 2048 }, { 1031, 2100 }, 
  { 1034, 2100 }, { 1039, 2100 }, { 1041, 2100 }, { 1049, 2100 }, 
  { 1052, 2240 }, { 1061, 2240 }, { 1063, 2304 }, { 1069, 2240 }, 
  { 1070, 2240 }, { 1074, 2240 }, { 1076, 2240 }, { 1077, 2240 }, 
  { 1081, 2240 }, { 1097, 2240 }, { 1108, 2240 }, { 1109, 2240 }, 
  { 1112, 2304 }, { 1114, 2304 }, { 1129, 2304 }, { 1132, 2304 }, 
  { 1149, 2520 }, { 1151, 2304 }, { 1163, 2520 }, { 1169, 2520 }, 
  { 1174, 2520 }, { 1175, 2520 }, { 1177, 2520 }, { 1181, 2520 }, 
  { 1187, 2520 }, { 1193, 2520 }, { 1198, 2520 }, { 1222, 2520 }, 
  { 1223, 2520 }, { 1229, 2520 }, { 1237, 2520 }, { 1238, 2520 }, 
  { 1251, 2520 }, { 1253, 2520 }, { 1259, 2520 }, { 1268, 2560 }, 
  { 1269, 2560 }, { 1286, 2592 }, { 1289, 2592 }, { 1307, 2880 }, 
  { 1315, 2688 }, { 1318, 2688 }, { 1319, 2688 }, { 1336, 2688 }, 
  { 1345, 2880 }, { 1363, 2880 }, { 1367, 2880 }, { 1381, 2880 }, 
  { 1382, 2880 }, { 1383, 2880 }, { 1385, 2880 }, { 1388, 2880 }, 
  { 1390, 2880 }, { 1391, 2880 }, { 1399, 2880 }, { 1401, 2880 }, 
  { 1410, 2880 }, { 1415, 2880 }, { 1418, 2880 }, { 1423, 2880 }, 
  { 1427, 2880 }, { 1432, 2880 }, { 1433, 2880 }, { 1436, 2880 }, 
  { 1437, 2880 }, { 1438, 2880 }, { 1439, 2880 }, { 1457, 3072 }, 
  { 1486, 3072 }, { 1487, 3072 }, { 1493, 3072 }, { 1497, 3072 }, 
  { 1498, 3072 }, { 1499, 3072 }, { 1503, 3072 }, { 1509, 3072 }, 
  { 1523, 3072 }, { 1529, 3072 }, { 1532, 3072 }, { 1551, 3240 }, 
  { 1571, 3240 }, { 1574, 3240 }, { 1578, 3240 }, { 1579, 3360 }, 
  { 1585, 3240 }, { 1594, 3240 }, { 1598, 3240 }, { 1605, 3240 }, 
  { 1609, 3240 }, { 1611, 3240 }, { 1614, 3240 }, { 1619, 3240 }, 
  { 1637, 3360 }, { 1646, 3360 }, { 1654, 3360 }, { 1657, 3360 }, 
  { 1658, 3360 }, { 1662, 3360 }, { 1663, 3360 }, { 1668, 3360 }, 
  { 1669, 3360 }, { 1670, 3360 }, { 1671, 3360 }, { 1678, 3360 }, 
  { 1693, 3584 }, { 1697, 3584 }, { 1698, 3584 }, { 1699, 3584 }, 
  { 1714, 3456 }, { 1726, 3584 }, { 1735, 3584 }, { 1739, 3584 }, 
  { 1759, 3584 }, { 1761, 3584 }, { 1774, 3584 }, { 1786, 3584 }, 
  { 1787, 3840 }, { 1789, 3584 }, { 1790, 3584 }, { 1795, 3840 }, 
  { 1797, 3840 }, { 1807, 3840 }, { 1819, 3840 }, { 1823, 3840 }, 
  { 1833, 3840 }, { 1837, 3840 }, { 1841, 3840 }, { 1857, 3780 }, 
  { 1867, 3840 }, { 1868, 3840 }, { 1877, 3840 }, { 1879, 3840 }, 
  { 1882, 3840 }, { 1883, 3840 }, { 1889, 3840 }, { 1902, 3840 }, 
  { 1907, 3840 }, { 1913, 3840 }, { 1915, 3840 }, { 1916, 4096 }, 
  { 1927, 4032 }, { 1929, 4096 }, { 1933, 4096 }, { 1934, 4096 }, 
  { 1939, 4032 }, { 1949, 4032 }, { 1969, 4096 }, { 1977, 4032 }, 
  { 1979, 4096 }, { 1981, 4096 }, { 1987, 4096 }, { 1993, 4032 }, 
  { 1994, 4032 }, { 1996, 4096 }, { 1997, 4096 }, { 2004, 4096 }, 
  { 2011, 4096 }, { 2021, 4096 }, { 2026, 4096 }, { 2027, 4096 }, 
  { 2033, 4096 }, { 2038, 4096 }, { 2039, 4096 }, { 2062, 4200 }, 
  { 2063, 4200 }, { 2068, 4200 }, { 2069, 4200 }, { 2073, 4200 }, 
  { 2078, 4200 }, { 2082, 4200 }, { 2083, 4200 }, { 2087, 4200 }, 
  { 2098, 4200 }, { 2099, 4200 }, { 2126, 4480 }, { 2127, 4480 }, 
  { 2137, 4480 }, { 2138, 4480 }, { 2141, 4480 }, { 2152, 4480 }, 
  { 2153, 4480 }, { 2154, 4480 }, { 2157, 4480 }, { 2162, 4480 }, 
  { 2171, 4480 }, { 2194, 4480 }, { 2203, 4480 }, { 2207, 4480 }, 
  { 2209, 4480 }, { 2213, 4480 }, { 2218, 4480 }, { 2219, 4480 }, 
  { 2228, 4480 }, { 2229, 4480 }, { 2239, 4480 }, { 2243, 4608 }, 
  { 2258, 4608 }, { 2264, 4608 }, { 2267, 4608 }, { 2293, 4608 }, 
  { 2298, 4608 }, { 2302, 4608 }, { 2303, 4608 }, { 2326, 5040 }, 
  { 2327, 5040 }, { 2333, 5040 }, { 2335, 5040 }, { 2338, 5040 }, 
  { 2339, 5040 }, { 2347, 5040 }, { 2348, 5040 }, { 2351, 5040 }, 
  { 2354, 5040 }, { 2361, 5040 }, { 2362, 5040 }, { 2363, 5040 }, 
  { 2367, 5040 }, { 2371, 5040 }, { 2374, 5040 }, { 2383, 5040 }, 
  { 2386, 5040 }, { 2389, 5040 }, { 2391, 5040 }, { 2393, 5040 }, 
  { 2395, 5040 }, { 2396, 5040 }, { 2397, 5040 }, { 2399, 5040 }, 
  { 2421, 5040 }, { 2423, 5040 }, { 2429, 5040 }, { 2444, 5040 }, 
  { 2446, 5040 }, { 2447, 5040 }, { 2458, 5040 }, { 2459, 5040 }, 
  { 2461, 5040 }, { 2467, 5040 }, { 2469, 5040 }, { 2473, 5040 }, 
  { 2474, 5040 }, { 2476, 5040 }, { 2477, 5040 }, { 2481, 5040 }, 
  { 2487, 5040 }, { 2491, 5040 }, { 2495, 5040 }, { 2503, 5040 }, 
  { 2505, 5040 }, { 2506, 5040 }, { 2513, 5040 }, { 2517, 5040 }, 
  { 2518, 5040 }, { 2531, 5120 }, { 2539, 5120 }, { 2547, 5120 }, 
  { 2571, 5184 }, { 2572, 5184 }, { 2578, 5184 }, { 2579, 5184 }, 
  { 2585, 5184 }, { 2589, 5184 }, { 2614, 5376 }, { 2621, 5376 }, 
  { 2633, 5376 }, { 2636, 5376 }, { 2638, 5376 }, { 2641, 5376 }, 
  { 2657, 5376 }, { 2659, 5376 }, { 2661, 5376 }, { 2663, 5376 }, 
  { 2671, 5376 }, { 2672, 5376 }, { 2677, 5376 }, { 2679, 5376 }, 
  { 2681, 5376 }, { 2683, 5376 }, { 2685, 5376 }, { 2687, 5376 }, 
  { 2726, 5760 }, { 2734, 5760 }, { 2741, 5760 }, { 2749, 5760 }, 
  { 2767, 5760 }, { 2773, 5760 }, { 2776, 5760 }, { 2777, 5760 }, 
  { 2782, 5760 }, { 2785, 5760 }, { 2797, 5760 }, { 2798, 5760 }, 
  { 2802, 5760 }, { 2803, 5760 }, { 2809, 5760 }, { 2819, 5760 }, 
  { 2823, 5760 }, { 2830, 5760 }, { 2833, 5760 }, { 2836, 5760 }, 
  { 2837, 5760 }, { 2839, 5760 }, { 2846, 5760 }, { 2853, 5760 }, 
  { 2854, 5760 }, { 2866, 5760 }, { 2867, 5760 }, { 2872, 5760 }, 
  { 2874, 5760 }, { 2876, 5760 }, { 2878, 5760 }, { 2879, 5760 }, 
  { 2893, 5880 }, { 2903, 5880 }, { 2909, 5880 }, { 2914, 5880 }, 
  { 2935, 5880 }, { 2957, 6144 }, { 2959, 6144 }, { 2963, 6144 }, 
  { 2972, 6144 }, { 2974, 6144 }, { 2986, 6144 }, { 2991, 6144 }, 
  { 2994, 6144 }, { 2995, 6144 }, { 2998, 6144 }, { 2999, 6144 }, 
  { 3006, 6144 }, { 3019, 6144 }, { 3023, 6144 }, { 3037, 6144 }, 
  { 3039, 6144 }, { 3043, 6144 }, { 3046, 6144 }, { 3047, 6144 }, 
  { 3055, 6144 }, { 3057, 6144 }, { 3058, 6144 }, { 3064, 6144 }, 
  { 3083, 6720 }, { 3093, 6720 }, { 3095, 6720 }, { 3113, 6720 }, 
  { 3117, 6720 }, { 3119, 6720 }, { 3123, 6720 }, { 3142, 6720 }, 
  { 3148, 6720 }, { 3149, 6720 }, { 3158, 6720 }, { 3167, 6720 }, 
  { 3173, 6720 }, { 3187, 6720 }, { 3188, 6720 }, { 3189, 6720 }, 
  { 3197, 6720 }, { 3207, 6720 }, { 3215, 6720 }, { 3217, 6720 }, 
  { 3218, 6720 }, { 3229, 6720 }, { 3231, 6720 }, { 3238, 6720 }, 
  { 3243, 6720 }, { 3269, 6720 }, { 3274, 6720 }, { 3291, 6720 }, 
  { 3292, 6720 }, { 3295, 6720 }, { 3308, 6720 }, { 3317, 6720 }, 
  { 3319, 6720 }, { 3323, 6720 }, { 3326, 6720 }, { 3327, 6720 }, 
  { 3337, 6720 }, { 3338, 6720 }, { 3340, 6720 }, { 3342, 6720 }, 
  { 3343, 6720 }, { 3347, 6720 }, { 3353, 6720 }, { 3356, 6720 }, 
  { 3359, 6720 }, { 3386, 7168 }, { 3387, 7168 }, { 3396, 7168 }, 
  { 3398, 6912 }, { 3401, 7168 }, { 3407, 7168 }, { 3413, 7168 }, 
  { 3419, 7168 }, { 3428, 7168 }, { 3447, 7168 }, { 3449, 7168 }, 
  { 3452, 7168 }, { 3461, 7168 }, { 3467, 7168 }, { 3470, 7168 }, 
  { 3478, 7168 }, { 3481, 7168 }, { 3487, 7168 }, { 3489, 7168 }, 
  { 3491, 7168 }, { 3493, 7168 }, { 3497, 7168 }, { 3499, 7168 }, 
  { 3507, 7168 }, { 3517, 7168 }, { 3518, 7168 }, { 3522, 7168 }, 
  { 3531, 7168 }, { 3541, 7168 }, { 3543, 7168 }, { 3545, 7168 }, 
  { 3548, 7168 }, { 3559, 7168 }, { 3561, 7168 }, { 3574, 7168 }, 
  { 3578, 7168 }, { 3579, 7168 }, { 3581, 7168 }, { 3583, 7168 }, 
  { 3590, 7680 }, { 3594, 7680 }, { 3595, 7560 }, { 3623, 7680 }, 
  { 3643, 7680 }, { 3646, 7680 }, { 3659, 7680 }, { 3669, 7680 }, 
  { 3671, 7680 }, { 3674, 7680 }, { 3677, 7680 }, { 3679, 7680 }, 
  { 3682, 7680 }, { 3687, 7680 }, { 3709, 7680 }, { 3711, 7680 }, 
  { 3713, 7680 }, { 3714, 7680 }, { 3719, 7680 }, { 3733, 7680 }, 
  { 3734, 7680 }, { 3736, 7680 }, { 3739, 7680 }, { 3754, 7680 }, 
  { 3758, 7680 }, { 3761, 7680 }, { 3764, 7680 }, { 3766, 7680 }, 
  { 3767, 7680 }, { 3769, 7680 }, { 3777, 7680 }, { 3778, 7680 }, 
  { 3779, 7680 }, { 3793, 7680 }, { 3803, 7680 }, { 3804, 7680 }, 
  { 3814, 7680 }, { 3817, 7680 }, { 3821, 7680 }, { 3826, 7680 }, 
  { 3830, 7680 }, { 3832, 7680 }, { 3833, 7680 }, { 3841, 8064 }, 
  { 3853, 8064 }, { 3858, 8064 }, { 3863, 8064 }, { 3898, 8064 }, 
  { 3899, 8064 }, { 3901, 8064 }, { 3911, 8064 }, { 3917, 8064 }, 
  { 3919, 8064 }, { 3921, 8064 }, { 3923, 8064 }, { 3931, 8064 }, 
  { 3935, 8064 }, { 3938, 8064 }, { 3945, 8064 }, { 3947, 8064 }, 
  { 3949, 8064 }, { 3954, 8064 }, { 3957, 8064 }, { 3958, 8064 }, 
  { 3959, 8064 }, { 3962, 8064 }, { 3967, 8064 }, { 3974, 8064 }, 
  { 3985, 8064 }, { 3986, 8064 }, { 3988, 8064 }, { 3989, 8064 }, 
  { 3992, 8064 }, { 3994, 8064 }, { 3995, 8064 }, { 4003, 8064 }, 
  { 4007, 8064 }, { 4008, 8064 }, { 4013, 8064 }, { 4021, 8064 }, 
  { 4022, 8064 }, { 4031, 8064 }, { 4035, 8192 }, { 4042, 8192 }, 
  { 4049, 8192 }, { 4052, 8192 }, { 4054, 8192 }, { 4066, 8192 }, 
  { 4073, 8192 }, { 4076, 8192 }, { 4078, 8192 }, { 4079, 8192 }, 
  { 4089, 8192 }, { 4091, 8192 },

{-1, -1}};


/* Compiler/Operating System dependent settings
   ============================================
*/

#if fft_NamesForUnusedParameters
#define UNUSED_PARAMETER(x) x
#else
#define UNUSED_PARAMETER(x)
#endif

/* Memory Management
   =================

   All variable sized storage is allocated/deallocated
   through malloc()/free(). There are no limitations due
   to static allocation of input size dependent variables.

   The buffers used in computations are allocated with
   alloca() so that the fft-object can be shared by multiple
   threads and no malloc/free is necessary for every single fft.
*/

#define MallocSize_shortMessage 200 /* a short text */

static void *Malloc(size_t length) {
  void *p;

  /* avoid mallocs of size below 32 bytes */
  if (length < 32)
    length = 32;

  p = malloc(length);
  if (p == NULL)
    fft_error("out of memory in malloc()");
  return p;
}

static void Free(void *p) {
  free(p);
}

#if fft_StackAllocWithAlloca
#define Alloca(length) alloca(length)
#define Dealloca(p)
#else
#define Alloca(length) malloc(length)
#define Dealloca(p)    free(p)
#endif

/* Error Messages
   ==============

   The global function or macro fft_error(msg) is called with
   the entire error message printed into a C-string which is
   either a constant or has been allocated with Malloc().
*/

#define strlen_int 20

static void Error(char *msg) {
  fft_error(msg);
}

static void Error_int(char *fmt, int x1) {
  char *msg;

  msg = (char *) Malloc(strlen(fmt) + strlen_int + 1);
  sprintf(msg, fmt, x1);
  fft_error(msg);  
  Free(msg);
}

static void Error_int2(char *fmt, int x1, int x2) {
  char *msg;

  msg = (char *) Malloc(strlen(fmt) + 2*strlen_int + 1);
  sprintf(msg, fmt, x1, x2);
  fft_error(msg);  
  Free(msg);
}

static void Error_int3(char *fmt, int x1, int x2, int x3) {
  char *msg;

  msg = (char *) Malloc(strlen(fmt) + 3*strlen_int + 1);
  sprintf(msg, fmt, x1, x2, x3);
  fft_error(msg);
  Free(msg);
}

/* Generic Arithmetics in the Base Field
   =====================================

   The arithmetics of the base field is parameterized in `fft.h'.
*/

/* for characteristic 0 */
#if !defined(Pi)
#define Pi 3.14159265358979323846264338328
#endif

/* abbreviations for the frequent operations */
#define set fft_set
#define add fft_add
#define sub fft_sub
#define mul fft_mul

/* declarations for scalars and vectors of scalars */
#define Scalar(x) fft_value (x)[fft_valuesPerScalar]
#define Vector(x) fft_value *(x)

/* access to a vector component */
#define at(vector, index) ((vector) + (index)*fft_valuesPerScalar)

/* (fft_value *) new_vector(N)
   delete_vector(x)
     create and delete a vector of length N

   (fft_value *) newa_vector(N)
   deletea_vector(x)
     allocate/deallocate a vector of length N from the stack.
*/     

static fft_value *new_vector(int N) {
  return 
    (fft_value *) Malloc(
      N * fft_valuesPerScalar * sizeof(fft_value)
    );
}

static void delete_vector(Vector(x)) {
  Free(x);
}

#define newa_vector(N) \
  (fft_value *)Alloca((N)*fft_valuesPerScalar*sizeof(fft_value))

#define deletea_vector(x)

/* rational(x, k, n)
     makes x = k/n for integers k, n and n > 0. The function
     issues an error if n = 0 in the base field.
     
   rootOfUnity(w, n, k)
     makes w a the k-th power of a fixed primitive n-th root
     of unity. The function issues an error if the base field
     does not contain the n-th roots of unity.
     
   rootsOfUnity(w, n)
     assigns all n-th roots of unity to w[0], w[1], .., w[n-1].
     The function computes w[2], .., w[n-1] by successive 
     multiplication with w[1]. Since in numerical computations
     errors accumulate, every max_rootsOfUnity_recurrence a new
     root of unity is computed from scratch. This maintains a
     balance between accuracy and speed.
*/

static void rational(Scalar(x), int k, int n) {
  if (! fft_rational(x, k, n))
    Error_int(
      "%d must be != 0 in the base field for FFT",
       n
    );
}

static void rootOfUnity(Scalar(w), int n, int k) {
  while (k < 0)
    k += n;
  k = k % n;

  if (! fft_rootOfUnity(w, n, k))
    Error_int(
      "base field must contain the %d-th roots of unity for FFT",
                                   n
    );
}

static void rootsOfUnity(Vector(w), int n) {
  int k;

  /* for char = p we assume exact arithmetics;
     Then we may safely compute the roots of unity
     by the trigonometric recurrence.
  */
  if (fft_char != 0) {
    rational(at(w, 0), 1, 1);
    rootOfUnity(at(w, 1), n, 1);
    for (k = 2; k < n; ++k)
      mul(at(w, k), at(w, k-1), at(w, 1));
    return;
  }

  /* for char = 0 and numerical arithmetics;
     Then there is a trade-off between speed (recurrence)
     and accuracy (always call sin/cos) for the roots of unity.
     Hence, we are to use (double) for the recurrence even if
     we compute the fft in (float).
  */
  if (fft_numerical) {

    double re_k = 1.0,
           im_k = 0.0,
           re_1 = cos(2.0*Pi/(double)n),
           im_1 = sin(2.0*Pi/(double)n);

    fft_setNumerical(at(w, 0), re_k, im_k);
    for (k = 1; k < n; ++k) {
      double re_k1 = re_k;

      re_k = re_k1 * re_1 - im_k * im_1;
      im_k = re_k1 * im_1 + im_k * re_1;
      fft_setNumerical(at(w, k), re_k, im_k);
    }
    return;

  }

  /* the general case; eval rootOfUnity() all the time */
  rational(at(w, 0), 1, 1);
  for (k = 1; k < n; ++k)
    rootOfUnity(at(w, k), n, k);
}

/* Permutations
   ============

   We implement permutations on {0..n-1} where n <= fft_max_perm.
   The permutations are stored in cycle notation so that
   in-place application is possible with a single register.
   In particular let s[0], s[1], .. store the first cycle
   until a negative value s[k] occurs, then -s[k]-1 is the
   last point of the first cycle. s[k+1] is the first point
   of the second cycle etc. until a cycle of length one occurs.

   Multiplication of permutations and inverses are best performed
   before calling new_perm, which converts to cycle structure.
   In particular,
     (s^-1)[ s[i] ] = i,       and
     (s*t)[ i ]     = t[s[i]]  for all i in [0..n-1].

   Applying permutation s to a vector x is identical to multiplying

     P(s) x = [ x[s[i]] : i in [0..n-1] ]

   where P(s) = [ delta_(s[i], j) ]_ij is the permutation matrix of s.
*/

/* fft_perm
     is an integer type which contains signed integers
     to store the image points of a permutation. The type
     may be (short int) if fft_max_perm < 2^15 - 1.

   fft_max_perm
     is a symbolic constant which limits the length of the
     maximal permutation possible (defined in fft.h).

   Apply_perm(r, N, perm, s, x)
     assignes x = (1_r (x) P(perm) (x) 1_t) . x for the permutation
     perm of [0..N-1) which has been created by new_perm().
        There is a macro version and a function version of this
     object. The macro version is to be prefered but since it
     the expansion text is considerably long the function version
     is supplied to be used if the compiler complains.

   (fft_perm *)new_perm(n, s)
   delete_perm(s)
     allocate/deallocate new permutations. new_perm first converts 
     into cycle form, delete_perm returns NULL.

   fprint_perm(out, s)
     fprints the permutation s (in cycles) to (FILE *)out. Note
     that the permutation printed is on {1..n} instead of {0..n-1}!
     This is for debugging purposes only.
*/

#if fft_hasLargeMacros /* from `fft.h' */

/* the function in condensed form */
#define Apply_perm(r, N, perm, s, x) \
  { int ir,is; fft_perm *t,i0,i; Vector(x0); Scalar(xi); Scalar(temp); \
  for (ir = 0; ir < r; ++ir) { for (is = 0; is < s; ++is) { \
  x0 = at(x, ir*(N)*s + is); t = perm; while (*t >= 0) { i0 = *(t++); \
  i = i0; set(xi, at(x0, i*s)); while ((*t) >= 0) { set(temp, xi); \
  i = *(t++); set(xi, at(x0, i*s)); set(at(x0, i*s), temp); } \
  set(temp, xi); i = -(*(t++))-1; set(xi, at(x0, i*s)); \
  set(at(x0, i*s), temp); set(at(x0, i0*s), xi); } } } }

#else

static void Apply_perm(int r, int N, fft_perm perm[], int s, Vector(x)) {
  int       ir, is;      /* counter for [0..r), [0..s) */
  fft_perm *t,                    /* runs through perm */
            i0,                /* first point in orbit */
            i;               /* current point in orbit */
  Vector(   x0);   /* base of (ir, is)-signal (step s) */
  Scalar(   xi);     /* x[ir*N*s + i*s + is] = x0[i*s] */
  Scalar(   temp);                 /* temporary buffer */

  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      /* permute the (ir, is)-part of x (thus we just need a 
         single register, otherwise it would be r, s dependent) 
      */
      x0 = at(x, ir*N*s + is);
      t  = perm;
      while (*t >= 0) {

        /* apply the cycle (t[0], t[1], .., t[r-2], -t[r-1]-1) 
           where t is being moved, i0 is the first point,
           i is the current point, xi is the value at i
        */
        i0 = *(t++);
        i  = i0;
        set(xi, at(x0, i*s));
        while ((*t) >= 0) {
          set(temp, xi);
          i = *(t++);
          set(xi, at(x0, i*s));
          set(at(x0, i*s), temp);
        }
        set(temp, xi);
        i = -(*(t++))-1;
        set(xi, at(x0, i*s));
        set(at(x0, i*s), temp);
        set(at(x0, i0*s), xi);
      }
    }
  }
}

#endif /* fft_hasLargeMacros */

/* make_perm(n1, s1, n, s)
     converts the permutation [s[0], .., s[n-1]] into
     cycle form [s1[0], .. s1[n1-1]] suitable for Apply_perm. 
     The array pointed to by s1 must have length at least n+1.
     This function is only used by new_perm().
*/

static void make_perm(int *n1, fft_perm *s1, int n, fft_perm *s) {
  int       nt;
  fft_perm  i0, i, ti;
  fft_perm *t;

  if (!( (1 <= n) && (n <= fft_max_perm) ))
    Error_int2(
      "n = %d must be in [0..%d) in make_perm()", 
           n,                fft_max_perm
    );

  t = (fft_perm *) Malloc( n * sizeof(fft_perm) );

  /* t = s^-1 and check if t is a permutation */
  for (i = 0; i < n; ++i)
    t[i] = -1;
  for (i = 0; i < n; ++i) {
    if ((s[i] < 0) || (s[i] >= n))
      Error_int3(
        "s[%d] = %d must be in [0..%d) in make_perm()",
           i,    s[i],             n
      );

    t[s[i]] = i;
  }
  for (i = 0; i < n; ++i)
    if (t[i] == -1)
      Error("s[] does not define a permutation in make_perm()");

  /* clear the cycles in t one after the other */
  nt = 0;
  i0 = 0;
  while (i0 < n) {

    /* find start of next non-trivial cycle */
    while ((i0 < n) && (t[i0] == i0))
      ++i0;

    /* clear the cycle (i0, t[i0], t[t[i0]], ..) */
    if (i0 < n) {
      i = i0;
      while (t[i] != i0) {
        s1[nt++] = i;
        ti      = t[i];
        t[i]    = i;
        i       = ti;
      } 
      s1[nt++] = -i-1;
      t[i]    = i;      
    }    
  }

  /* add end mark */
  s1[nt++] = -1;
  *n1 = nt;

  Free(t);
}

static fft_perm *new_perm(int n, fft_perm *s) {
  fft_perm *s0, *s1,
            n1, i;

  /* copy s into s1 with space for the end mark */
  s1 = (fft_perm *) Malloc( (n+1) * sizeof(fft_perm) );
  make_perm(&n1, s1,  n, s);

  /* copy into a new memory region of matching size */
  s0 = (fft_perm *) Malloc( n1 * sizeof(fft_perm) );
  for (i = 0; i < n1; ++i)
    s0[i] = s1[i];

  Free(s1);
  return s0;
}

static void delete_perm(fft_perm *s) {
  Free(s);
}

/* for debugging only.. */

static void fprint_perm(FILE *out, fft_perm *s) {
  if (s[0] < 0) {
    fprintf(out, "()");
    return;
  }
  while (s[0] >= 0) {
    fprintf(out, "(%d", s[0] +1);
    while (s[1] >= 0) {
      ++s;
      fprintf(out, ",%d", s[0] +1);
    }
    ++s;
    fprintf(out, ",%d)", -s[0]-1 +1);
    ++s;
  }
}

/* ..for debugging only */

/* Number Theory
   =============

   A few tools from number theory are needed for the FFT.
   We implement them in (int), assuming (int) to be at
   least 32 bits. All functions, except the most primitive,
   are designed for maximum reliability, rather than for speed.
   The usable range of numbers is limited to 
     -2^30 .. 2^30
   to avoid silent overflow in additions.
*/

#define checkInteger(func, x) \
  if (((x) < -1073741824) || ((x) > 1073741824)) { \
    Error_int("integer range violation for %d in " func, (x)); }

#define checkModulus(func, x) \
  if (((x) < 1) || ((x) > 1073741824)) { \
    Error_int("modulus range violation for %d in " func, (x)); }

/* mod(x, m)
   div(x, m)
     compute numbers such that x = div(x, m)*m + mod(x, m) where
     0 <= mod(x, m) < m.
*/

#define mod(x, m) ((x) < 0 ? ((x)+(m))%(m) : (x)%(m))
#define div(x, m) ((x)/(m))

/* gcd(a, b)
   gcdex(a, b, &s, &t)
     compute the greatest common divisor of a and b. gcdex() also
     returns s, t such that gcd(a, b) = s*a + t*b. Note that
       (1) gcd(0, 0) = 0
       (2) gcd(a, b) > 0 for a, b != 0.
*/

static int gcd(int a, int b) {
  int c;

  checkInteger("gcd", a);
  checkInteger("gcd", b);

  while (b != 0) {
    c = mod(a, b);
    a = b;
    b = c;
  }
  return a;
}

static int gcdex(int a, int b, int *s, int *t) {
  int sa,sb, a0,a1, s0,s1, t0,t1, q, x;

  checkInteger("gcdex", a);
  checkInteger("gcdex", b);

  if (a < 0) { sa = -1; a = -a; } else { sa = 1; }
  if (b < 0) { sb = -1; b = -b; } else { sb = 1; }

  /* s a + t b = gcd(a, b) */
  a0 = a;  a1 = b;
  s0 = 1;  s1 = 0;
  t0 = 0;  t1 = 1;
  while (a1 != 0) {
    q = div(a0, a1);
    x = a0 - a1*q; a0 = a1; a1 = x;
    x = s0 - s1*q; s0 = s1; s1 = x;
    x = t0 - t1*q; t0 = t1; t1 = x;
  }
  *s = sa * s0;
  *t = sb * t0;
  if ((*s)*a + (*t)*b == a0)
    return a0;

  Error("overflow in gcdex()");
  return -1;
}

/* addMod(x, y, m)
   mulMod(x, y, m)
   powerMod(x, k, m)
     compute mod(x + y, m), mod(x * y, m), and mod(x^k, m) 
     for m > 0. If k < 0 then modular inverse are computed.
*/

#define addMod(x, y, m) mod((x) + (y), m)

#define mulMod(x, y, m) \
  (((x) < 32768) && ((y) < 32768) ? \
    mod((x)*(y), m) : mulMod32(x, y, m))

static int mulMod32(int x, int y, int m) {
  int              u;
  static const int M = 32768;

  checkInteger("mulMod32", x);
  checkInteger("mulMod32", y);
  checkModulus("mulMod32", m);

  x = mod(x, m);
  y = mod(y, m);
  if ((x == 0) || (y == 0))
    return 0;
  if (x == 1)
    return y;
  if (y == 1)
    return x;

  /* 0 <= x, y < m. Now m^2 < 2^30 <==> m < 2^15,
     so this is the easy case.
  */
  if ((x < M) && (y < M))
    return mod(x * y, m);

  /* shift-and-add */
  u = mulMod(div(x, 2), y, m);
  u = mod((u + u) % m, m);
  if (mod(x, 2) == 0)
    return u;
  else
    return mod((u + y) % m, m);
}

static int powerMod(int x, int k, int m) {
  int u, v;

  checkInteger("powerMod", x);
  checkInteger("powerMod", k);
  checkModulus("powerMod", m);

  x = mod(x, m);
  if (m == 1) 
    return 0;
  if (k == 0)
    return 1;
  if (k == 1)
    return x;

  /* reduce modular inverses to gcdex and powerMod */
  if (k < 0) {
    if (gcdex(x, m, &u, &v) != 1)
      Error_int3(
        "powerMod(%d, %d, %d) does not exist", 
                  x,  k,  m
      );
    return powerMod(u, -k, m);
  }

  /* recursive square-and-multiply */
  u = powerMod(x, div(k, 2), m);
  u = mulMod(u, u, m);
  if (mod(k, 2) == 0)
    return u;
  else
    return mulMod(u, x, m);
}

/* primes[0], .., primes[n_primes-1]
     is the list of the first n_primes prime numbers.

   fft_factor(x, &n, &(p[0]), &(e[0]))
     computes the prime factorization of x > 0 and stores it as
       x = p[0]^e[0] * .. * p[n-1]^e[n-1].
     The symbolic constant max_factor is defined such that 
     n <= max_factor holds for all x in [1..2^31-1].

   primeQ(x)
     determines whether x is prime.
*/

#define           n_primes 168
static int primes[n_primes] = {
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
  73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 
  157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 
  239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 
  331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 
  421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 
  509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 
  613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 
  709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 
  821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 
  919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997
};

void fft_factor(int x, int *n, int p[], int e[]) {
  int ip;

  checkInteger("fft_factor", x);
  if (x <= 0)
    Error_int("x = %d must be positive in fft_factor()", x);

  /* divide off primes[ip] */
  ip = 0;
  *n = 0;
  while ((x > 1) && (ip < n_primes)) {
    p[*n] = primes[ip];
    e[*n] = 0;
    while (mod(x, p[*n]) == 0) {
      x /= p[*n];
      ++(e[*n]);
    }
    if (e[*n] > 0)
      ++(*n);
    ++ip;
  }

  /* divide off p[*n] until p[*n]^2 > x */
  p[*n] = primes[n_primes-1]+2;
  while (x > 1) {
    while ((p[*n]*p[*n] <= x) && (mod(x, p[*n]) != 0))
      p[*n] += 2;
    e[*n] = 0;
    while (mod(x, p[*n]) == 0) {
      x /= p[*n];
      ++(e[*n]);
    }
    if (e[*n] > 0) {
      ++(*n);
      p[*n] = p[*n - 1] + 2;
    } else {
      p[*n] += 2;
    }
  }
}

static int primeQ(int x) {
  int n, p[fft_max_factors], e[fft_max_factors];

  checkInteger("primeQ", x);
  if (x < 2)
    return 0;
  fft_factor(x, &n, p, e);
  return ((n == 1) && (e[0] == 1));
}

/* generatorPrimeField(p)
     given a prime p, this function determines the smallest integer
     q in [2..p-1] such that {mod(q^k, p) | k in {0..p-2}} = {1..p-1}.
     They name stems from the fact that q*Z is a generator of the 
     multiplicative group of the prime field F_p = Z/p*Z, where Z 
     denotes the integers.
*/

static int generatorPrimeField(int p) {
  int  q, n,T[fft_max_factors],dummy[fft_max_factors], i;

  /* Lipson, Chapter IX.1.3.
     (1) The density of primitive elements in F_p is relatively high,
         namely 3/pi^2 averaged over all primes p. Hence, we can search F_p.
     (2) q in F_p is primitive iff
         mod(q^((p-1)/t), p) != 1 for all primes t dividing p-1.
  */ 

  /* check p */
  checkInteger("generatorPrimeField", p);
  if (! primeQ(p))
    Error_int("p = %d must be prime in generatorPrimeField()", p);
  if (p == 2)
    return 1;

  /* {T[0], .. T[n-1]} := { (p-1)/t | t prime, t | (p-1) } */
  fft_factor(p-1, &n, T, dummy);
  for (i = 0; i < n; ++i)
    T[i] = (p-1)/T[i];

  /* try all q = 2..p-1 in turn and test primitivity */
  for (q = 2; q < p; ++q) {
    i = 0;
    while ((i < n) && (powerMod(q, T[i], p) != 1))
      ++i;
    if (i == n)
      return q;
  }

  Error_int("generatorPrimeField(%d) failed", p);
  return -1;
}

/* Generic-FFT
   ===========
*/

void fft_apply_scalar(int r, fft_t *F, int s, Vector(x), Scalar(a)) {
  int    i;                     /* counter for [0..r N s) */
  Vector(x1);                        /* pointer through x */

  x1 = x;
  for (i = 0; i < r*(F->N)*s; ++i) {
    mul(x1, x1, a);
    x1 = at(x1, 1);
  }  
}

void fft_apply_reverse(int r, fft_t *F, int s, Vector(x)) {
  int    N,                              /* length of FFT */
         ir, k, is; /* counter for [0..r), [0..N), [0..s) */
  Vector(x1);                        /* pointer through x */
  Vector(x2);                        /* pointer through x */
  Scalar(temp);                 /* temporary for swapping */

  /* x = (1_r (x) R (x) 1_s) x where R(0) = 0, R(k) = N-k */ 
  N = F->N;
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {
      x1 = at(x, ir*N*s +     1*s + is);
      x2 = at(x, ir*N*s + (N-1)*s + is);
      for (k = 1; k < N-k; ++k) {
        set(temp, x1); set(x1, x2); set(x2, temp);
        x1 = at(x1,  s);
        x2 = at(x2, -s);
      }
    }
  }
}

/* Null-FFT 
   ========

   For debugging and profiling we supply a fake FFT.
   In addition fft_new_null() creates a new (fft_t) structure to
   be used by other methods.
*/

static void apply_null(
  int           UNUSED_PARAMETER(r), 
  struct fft_s *UNUSED_PARAMETER(F), 
  int           UNUSED_PARAMETER(s), 
  fft_value    *UNUSED_PARAMETER(x)
) {
  return;
}

static void delete_null(struct fft_s *F) {
  Free(F);
}

fft_t *fft_new_null(int N) {
  fft_t *F;
  Scalar(dummy);
  char  *msg;

  msg = fft_applicable(fft_null, N, 0, NULL);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }

  F            = (fft_t *) Malloc(sizeof(fft_t));
  F->type      = fft_null;
  F->N         = N;
  F->apply     = apply_null;
  F->deleteObj = delete_null;

  /* build 1/N, w(N) and possibly 1/Sqrt[N] */
  rational(F->inverseN, 1, N);
#if (fft_char == 0)
  fft_rational(F->inverseSqrtN, 1.0, sqrt((double) N));
#endif
  rootOfUnity(dummy, N, 1);

  return F;
}

/* Direct FFT (Matrix Multiplication)
   ==================================
*/

static void apply_direct(int r, struct fft_s *F, int s, Vector(x)) {
  int    N,                           /* length */
         ir, is,  /* counter for [0..r), [0..s) */
         i, j, k,         /* counter for [0..N) */
         ij;                       /* i*j mod N */
  Vector(w);             /* N-th roots of unity */
  Vector(x0);      /* pointer for signal values */
  Vector(y0);      /* pointer for signal values */
  Scalar(temp);  /* temporary to accumulate sum */
  Vector(y) = newa_vector(F->N);      /* buffer */

  N = F->N;
  w = F->priv.direct.rootOfUnity;

  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      /* [y[k] : k in [0..N)] = [x[ir*N*s + k*s + is] : k in [0..N)] */
      x0 = at(x, ir*N*s + is);
      y0 = y;
      for (k = 0; k < N; ++k) {
        set(y0, x0);
        x0 = at(x0, s);
        y0 = at(y0, 1);
      }

      /* x[ir*N*s + i*s + is] = Sum(w(N)^(i*j) y[j] : j in [0..N)) */
      for (i = 0; i < N; ++i) {
        x0  = at(x, ir*N*s + i*s + is);
        y0  = y;

        ij  = 0;
        set(x0, y0); /* w(N)^ij = 1 */
        y0  = at(y0, 1);
        ij += i;
        for (j = 1; j < N; ++j) {
          mul(temp, y0, at(w, ij)); 
          add(x0, x0, temp);
          y0 = at(y0, 1);
          ij = addMod(ij, i, N);
        }
      }
    }
  }
  deletea_vector(y);
}

static void delete_direct(struct fft_s *F) {
  delete_vector(F->priv.direct.rootOfUnity);
  delete_null(F);
}

fft_t *fft_new_direct(int N) {
  fft_t *F;
  char  *msg;

  msg = fft_applicable(fft_direct, N, 0, NULL);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }

  /* create F */
  F            = fft_new_null(N);
  F->type      = fft_direct;
  F->N         = N;
  F->apply     = apply_direct;
  F->deleteObj = delete_direct;

  /* precompute N-th roots of unity */
  F->priv.direct.rootOfUnity = new_vector(N);
  rootsOfUnity(F->priv.direct.rootOfUnity, N);
  return F;
}

/* Small-FFT
   =========

   For N <= fft_max_small we supply a straight-line implementation
   of the FFT(N). These functions are the basis of most other FFT's.
   If the small FFTs are modified or extended make sure
     1. `fft.h':fft_max_small is corrected.
     2. The number of constants F->priv.small.c used does not
        exceed the number allocated in fft_new_small().
     3. The restrictions imposed by using other roots of unity and
        rationals are known in fft_applicable().
*/

static void apply_small1(
  int           UNUSED_PARAMETER(r), 
  struct fft_s *UNUSED_PARAMETER(F), 
  int           UNUSED_PARAMETER(s), 
  fft_value    *UNUSED_PARAMETER(x)
) {
  return;
}

/* FFT on 2 points
   ---------------

   The FFT on 2 points is trivial.
   The algorithm can be used iff -1 != 1.
*/

static void apply_small2(
  int                            r, 
  struct fft_s *UNUSED_PARAMETER(F), 
  int                            s, 
  Vector(                        x )
) {
  int    ir, is;
  Vector(x0);
  Vector(x1);
  Scalar(t0);

  x0 = at(x, 0*s);
  x1 = at(x, 1*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      set(t0, x0);
      add(x0, t0, x1);
      sub(x1, t0, x1);

      x0 = at(x0, 1);
      x1 = at(x1, 1);
    }
    x0 = at(x0, -s + 2*s);
    x1 = at(x1, -s + 2*s);
  }
}

/* FFT on 3 points
   ---------------

   We take the algorithm form Nussbaumer, 1982, sect. 5.5.2.,
   modifying u -> -2 pi/3. We generalize the algorithm presented
   there to arbitrary base fields by converting the constants

     Cos[u] - 1 -> c1 = 3 * w[6]^3 / 2
     I Sin[u]   -> c2 = (w[6]^4 + w[6]^5) / 2

   The algorithm can be used iff 1/2, 1/3 and w(6) exists 
   in the base field.
*/

static void init_small3(Vector(c)) {
  Scalar(w); Scalar(s);
  
  rational(at(c, 0), -3, 2);

  rootOfUnity(at(c, 1), 6, 4);
  rootOfUnity(w,       6, 5);
  add(at(c, 1), at(c, 1), w);
  rational(s, 1, 2);
  mul(at(c, 1), at(c, 1), s);  
}

static void apply_small3(int r, struct fft_s *F, int s, Vector(x)) {
  int    ir, is;
  Vector(c);
  Vector(x0);
  Vector(x1);
  Vector(x2);
  Scalar(t1);
  Scalar(m0); Scalar(m1); Scalar(m2);
  Scalar(s1);

  /* Rader */
  c  = F->priv.small.c;
  x0 = at(x, 0*s);
  x1 = at(x, 1*s);
  x2 = at(x, 2*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      add(t1, x1, x2);
      add(m0, x0, t1);
      mul(m1, at(c,0), t1);
      sub(m2, x2, x1);
      mul(m2, at(c,1), m2);
      add(s1, m0, m1);
      set(x0, m0);
      add(x1, s1, m2);
      sub(x2, s1, m2);

      x0 = at(x0, 1);
      x1 = at(x1, 1);
      x2 = at(x2, 1);
    }
    x0 = at(x0, -s + 3*s);
    x1 = at(x1, -s + 3*s);
    x2 = at(x2, -s + 3*s);
  }
}

/* FFT on 4 points
   ---------------

   We implement the algorithm from Nussbaumer, 1982, sect. 5.5.3,
   modifying j -> -j. 
*/

static void init_small4(Vector(c)) {
  rootOfUnity(at(c, 0), 4, 3); /* w(4)^3 = -I */
}

static void apply_small4(int r, struct fft_s *F, int s, Vector(x)) {
  int    ir, is;
  Vector(c);
  Vector(x0);
  Vector(x1);
  Vector(x2);
  Vector(x3);
  Scalar(t1); Scalar(t2);
  Scalar(m0); Scalar(m1); Scalar(m2); Scalar(m3);

  c  = F->priv.small.c;
  x0 = at(x, 0*s);
  x1 = at(x, 1*s);
  x2 = at(x, 2*s);
  x3 = at(x, 3*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      add(t1, x0, x2);
      add(t2, x1, x3);
      add(m0, t1, t2);
      sub(m1, t1, t2);
      sub(m2, x0, x2);
      sub(m3, x3, x1);
      mul(m3, m3, at(c, 0));
      set(x0, m0);
      add(x1, m2, m3);
      set(x2, m1);
      sub(x3, m2, m3);

      x0 = at(x0, 1);
      x1 = at(x1, 1);
      x2 = at(x2, 1);
      x3 = at(x3, 1);
    }
    x0 = at(x0, -s + 4*s);
    x1 = at(x1, -s + 4*s);
    x2 = at(x2, -s + 4*s);
    x3 = at(x3, -s + 4*s);
  }
}

/* FFT on 5 points
   ---------------

   We implement the algorithm from Nussbaumer, 1982, sect. 5.5.4,
   modifying u -> -2 Pi/5 and restating the trigonometric constants
   as quantities in the 5-th roots of unity. In addition -1 = w(2)
   is needed, so the algorithm is applicable iff 1/5, 1/4, and w(10)
   exist in the base field.
   
   (Cos[u] + Cos[2 u])/2 - 1 -> c0 = -5/4
   (Cos[u] - Cos[2 u])/2     -> c1 = -1/2 (1/2 + w[5]^2 + w[5]^3)
   -I Sin[u]                 -> c2 = (1 + 2 w[5] + w[5]^2 + w[5]^3)/2
   -I (Sin[u] + Sin[2 u])    -> c3 = 1/2 + w[5] + w[5]^2
   I (Sin[u] - Sin[2 u])     -> c4 = -(1/2 + w[5] + w[5]^3)
*/

static void init_small5(Vector(c)) {
  Scalar(w); Scalar(s);
  
  rational(at(c, 0), -5, 4);

  rational(at(c, 1), 1, 2);
  rootOfUnity(w, 5, 2); 
  add(at(c, 1), at(c, 1), w);
  rootOfUnity(w, 5, 3); 
  add(at(c, 1), at(c, 1), w);
  rational(s, -1, 2);   
  mul(at(c, 1), at(c, 1), s);

  rational(at(c, 2), 1, 1);
  rational(s, 2, 1); 
  rootOfUnity(w, 5, 1); 
  mul(s, s, w);
  add(at(c, 2), at(c, 2), s);
  rootOfUnity(w, 5, 2);
  add(at(c, 2), at(c, 2), w);
  rootOfUnity(w, 5, 3);
  add(at(c, 2), at(c, 2), w);
  rational(s, 1, 2);
  mul(at(c, 2), at(c, 2), s);

  rational(at(c, 3), 1, 2);
  rootOfUnity(w, 5, 1);
  add(at(c, 3), at(c, 3), w);
  rootOfUnity(w, 5, 2);
  add(at(c, 3), at(c, 3), w);
   
  rational(at(c, 4), 1, 2);
  rootOfUnity(w, 5, 1);
  add(at(c, 4), at(c, 4), w);
  rootOfUnity(w, 5, 3);
  add(at(c, 4), at(c, 4), w);
  rational(s, -1, 1);
  mul(at(c, 4), at(c, 4), s);
}

static void apply_small5(int r, struct fft_s *F, int s, Vector(x)) {
  int    ir, is;
  Vector(c);
  Vector(x0); Vector(x1); Vector(x2); Vector(x3); Vector(x4);
  Scalar(t1); Scalar(t2); Scalar(t3); Scalar(t4); Scalar(t5);
  Scalar(m0); Scalar(m1); Scalar(m2); 
    Scalar(m3); Scalar(m4); Scalar(m5);
  Scalar(s1); Scalar(s2); Scalar(s3); Scalar(s4); Scalar(s5);

  c  = F->priv.small.c;
  x0 = at(x, 0*s);
  x1 = at(x, 1*s);
  x2 = at(x, 2*s);
  x3 = at(x, 3*s);
  x4 = at(x, 4*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      add(t1, x1, x4);
      add(t2, x2, x3);
      sub(t3, x1, x4);
      sub(t4, x3, x2);
      add(t5, t1, t2);
      add(m0, x0, t5);
      mul(m1, t5, at(c, 0));
      sub(m2, t1, t2);
      mul(m2, m2, at(c, 1));
      add(m3, t3, t4);
      mul(m3, m3, at(c, 2));
      mul(m4, t4, at(c, 3));
      mul(m5, t3, at(c, 4));
      add(s1, m0, m1);
      add(s2, s1, m2);
      sub(s3, m3, m4);
      sub(s4, s1, m2);
      add(s5, m3, m5);
      set(x0, m0);
      add(x1, s2, s3);
      add(x2, s4, s5);
      sub(x3, s4, s5);
      sub(x4, s2, s3);

      x0 = at(x0, 1);
      x1 = at(x1, 1);
      x2 = at(x2, 1);
      x3 = at(x3, 1);
      x4 = at(x4, 1);
    }
    x0 = at(x0, -s + 5*s);
    x1 = at(x1, -s + 5*s);
    x2 = at(x2, -s + 5*s);
    x3 = at(x3, -s + 5*s);
    x4 = at(x4, -s + 5*s);
  }
}

/* FFT on 6 points
   ---------------

   With the Good-Thomas FFT we find

     DFT[6] = 
       perm[{0, 4, 2, 3, 1, 5}]
       tensor[DFT[2], DFT[3]] 
       perm[{0, 2, 4, 3, 5, 1}].
*/

static void init_small6(Vector(c)) { 
  init_small3(c);
}

static void apply_small6(int r, struct fft_s *F, int s, Vector(x)) {
  int    ir, is;
  Vector(c);
  Vector(x0); Vector(x1); 
  Vector(x2); Vector(x3); 
  Vector(x4); Vector(x5);
  Scalar(t1); Scalar(s1);
  Scalar(m0); Scalar(m1); Scalar(m2);
 
  c  = F->priv.small.c;
  x0 = at(x, 0*s);  x1 = at(x, 1*s);
  x2 = at(x, 2*s);  x3 = at(x, 3*s);
  x4 = at(x, 4*s);  x5 = at(x, 5*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      /* x[0,2,4] = DFT(3) x[0,2,4]; 
         x[3,5,1] = DFT(3) x[3,5,1];
      */
      add(t1, x2, x4);
      add(m0, x0, t1);
      mul(m1, at(c,0), t1);
      sub(m2, x4, x2);
      mul(m2, at(c,1), m2);
      add(s1, m0, m1);
      set(x0, m0);
      add(x2, s1, m2);
      sub(x4, s1, m2);

      add(t1, x5, x1);
      add(m0, x3, t1);
      mul(m1, at(c,0), t1);
      sub(m2, x1, x5);
      mul(m2, at(c,1), m2);
      add(s1, m0, m1);
      set(x3, m0);
      add(x5, s1, m2);
      sub(x1, s1, m2);

      /* x[0,3] = DFT(2) x[0,3]; 
         x[4,1] = DFT(2) x[2,5];
         x[2,5] = DFT(2) x[4,1]; 
      */
      set(m0, x0);
      add(x0, m0, x3);
      sub(x3, m0, x3);

      set(m0, x4);
      set(m1, x1);
      add(x4, x2, x5);
      sub(x1, x2, x5);
      
      add(x2, m0, m1);
      sub(x5, m0, m1);

      x0 = at(x0, 1);  x1 = at(x1, 1); 
      x2 = at(x2, 1);  x3 = at(x3, 1);
      x4 = at(x4, 1);  x5 = at(x5, 1);
    }
    x0 = at(x0, -s + 6*s);  x1 = at(x1, -s + 6*s);
    x2 = at(x2, -s + 6*s);  x3 = at(x3, -s + 6*s);
    x4 = at(x4, -s + 6*s);  x5 = at(x5, -s + 6*s);
  }
}

/* FFT on 7 points
   ---------------

   We implement the algorithm from Nussbaumer, 1982, sect. 5.5.5,
   modifying u -> -2 Pi/7 and restating the trigonometric constants
   as quantities in the 7-th roots of unity. In addition -1 = w(2)
   is needed, so the algorithm is applicable iff 1/7, 1/6, and w(14)
   exist in the base field.

   (Cos[u] + Cos[2 u] + Cos[3 u])/3 - 1  -> c0
   (2 Cos[u] - Cos[2 u] - Cos[3 u])/3    -> c1
   (Cos[u] - 2 Cos[2 u] + Cos[3 u])/3    -> c2
   -(Cos[u] + Cos[2 u] - 2 Cos[3 u])/3   -> c3
   -I (Sin[u] + Sin[2 u] - Sin[3 u])/3   -> c4
   I (2 Sin[u] - Sin[2 u] + Sin[3 u])/3  -> c5
   I (Sin[u] - 2 Sin[2 u] - Sin[3 u])/3  -> c6
   -I (Sin[u] + Sin[2 u] + 2 Sin[3 u])/3 -> c7
      
   To avoid computations of the form x = -y - z we redefine 
   s0 -> -s0, s1 -> -s1, s2 -> -s2, t13 -> -t13, t14 -> -t14.
   This has fliped the sign in constants c3 and c7. Furthermore,
   we rearrange the assignments a little bit to seperate the
   t-, m-, and s-layers. The constants are

   c0 = -7/6
   c1 = (-2 - 3 w[7]^2 - 3 w[7]^3 - 3 w[7]^4 - 3 w[7]^5)/6
   c2 = (-1 - 3 w[7]^2 - 3 w[7]^5)/6
   c3 = (1 + 3 w[7]^3 + 3 w[7]^4)/6
   c4 = (1 + 2 w[7] + 2 w[7]^2 + 2 w[7]^4)/6
   c5 = (-2 - 4 w[7] - w[7]^2 - 3 w[7]^3 - w[7]^4 - 3 w[7]^5)/6
   c6 = (-1 - 2 w[7] + w[7]^2 - 2 w[7]^4 - 3 w[7]^5)/6
   c7 = (1 + 2 w[7] + 2 w[7]^2 + 3 w[7]^3 - w[7]^4)/6
*/

static void init_small7(Vector(c)) {
  int    k, i;
  Scalar(s); 
  Scalar(w);
  static int coeffsTimes6[8][6] = {
    { -7,  0,  0,  0,  0,  0 }, 
    { -2,  0, -3, -3, -3, -3 }, 
    { -1,  0, -3,  0,  0, -3 }, 
    {  1,  0,  0,  3,  3,  0 }, 
    {  1,  2,  2,  0,  2,  0 }, 
    { -2, -4, -1, -3, -1, -3 }, 
    { -1, -2,  1,  0, -2, -3 }, 
    {  1,  2,  2,  3, -1,  0 }
  };
  
  rational(at(c, 0), -7, 6);
  for (k = 1; k < 8; ++k) {
    rational(at(c, k), coeffsTimes6[k][0], 6);
    for (i = 1; i < 6; ++i) {
      rational(s, coeffsTimes6[k][i], 6);
      rootOfUnity(w, 7, i);
      mul(s, s, w);
      add(at(c, k), at(c, k), s);
    }
  }
}

static void apply_small7(int r, struct fft_s *F, int s, Vector(x)) { 
  int ir, is;
  Vector(c);
  Vector(x0);  Vector(x1);  Vector(x2);  Vector(x3);  Vector(x4); 
  Vector(x5);  Vector(x6);
  Scalar(t1);  Scalar(t2);  Scalar(t3);  Scalar(t4);  Scalar(t5);
  Scalar(t6);  Scalar(t7);  Scalar(t8);  Scalar(t9);  Scalar(t10); 
  Scalar(t11); Scalar(t12); Scalar(t13); Scalar(t14);
  Scalar(m0);  Scalar(m1);  Scalar(m2);  Scalar(m3);  Scalar(m4); 
  Scalar(m5);  Scalar(m6);  Scalar(m7);  Scalar(m8);
  Scalar(s0);  Scalar(s1);  Scalar(s2);  Scalar(s3);  Scalar(s4); 
  Scalar(s5);  Scalar(s6);  Scalar(s7);  Scalar(s8);  Scalar(s9); 
  Scalar(s10);

  c  = F->priv.small.c;
  x0 = at(x, 0*s);  x1 = at(x, 1*s);
  x2 = at(x, 2*s);  x3 = at(x, 3*s);
  x4 = at(x, 4*s);  x5 = at(x, 5*s);
  x6 = at(x, 6*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      add(t1, x1, x6);
      add(t2, x2, x5);
      add(t3, x3, x4);
      add(t4, t1, t2); add(t4, t4, t3);
      sub(t5, x1, x6);
      sub(t6, x2, x5);
      sub(t7, x4, x3);
      sub(t8, t1, t3);
      sub(t9, t3, t2);
      add(t10, t5, t6); add(t10, t10, t7);
      sub(t11, t7, t5);
      sub(t12, t6, t7);
      add(t13, t8, t9);
      add(t14, t11, t12);
      
      add(m0, x0, t4);    
      mul(m1, t4,  at(c, 0));
      mul(m2, t8,  at(c, 1));
      mul(m3, t9,  at(c, 2));
      mul(m4, t13, at(c, 3));
      mul(m5, t10, at(c, 4));
      mul(m6, t11, at(c, 5));
      mul(m7, t12, at(c, 6));
      mul(m8, t14, at(c, 7));

      add(s0, m2, m3);
      add(s1, m2, m4);
      add(s2, m6, m7);
      add(s3, m6, m8);
      add(s4, m0, m1);
      add(s5, s4, s0);
      sub(s6, s4, s1);
      sub(s7, s4, s0); add(s7, s7, s1);
      add(s8, m5, s2);
      sub(s9, m5, s3);
      sub(s10, m5, s2); add(s10, s10, s3);

      set(x0, m0);
      add(x1, s5, s8);
      add(x2, s6, s9);
      sub(x3, s7, s10);
      add(x4, s7, s10);
      sub(x5, s6, s9);
      sub(x6, s5, s8);

      x0 = at(x0, 1);  x1 = at(x1, 1);
      x2 = at(x2, 1);  x3 = at(x3, 1);
      x4 = at(x4, 1);  x5 = at(x5, 1);
      x6 = at(x6, 1);
    }
    x0 = at(x0, -s + 7*s);  x1 = at(x1, -s + 7*s);
    x2 = at(x2, -s + 7*s);  x3 = at(x3, -s + 7*s);
    x4 = at(x4, -s + 7*s);  x5 = at(x5, -s + 7*s);
    x6 = at(x6, -s + 7*s);
  }
}

/* FFT on 8 points
   ---------------

   We implement the algorithm from Nussbaumer, 1982, sect. 5.5.6,
   modifying u -> -2 Pi/8 and j -> -j. The algorithm is applicable
   iff the base field contains 1/8, w(8). The constants are
   
   Cos[u]       -> c0 = 1/2 (w[8] + w[8]^7)
   (-I)         -> c1 = w[8]^6
   -(-I) Sin[u] -> c2 = 1/2 (w[8] - w[8]^7)
*/

static void init_small8(Vector(c)) { 
  Scalar(s); Scalar(w);
  
  rootOfUnity(at(c, 0), 8, 1);
  rootOfUnity(w, 8, 7);
  add(at(c, 0), at(c, 0), w);
  rational(s, 1, 2);
  mul(at(c, 0), at(c, 0), s);
  
  rootOfUnity(at(c, 1), 8, 6);
  
  rootOfUnity(at(c, 2), 8, 1);
  rootOfUnity(w, 8, 7);
  sub(at(c, 2), at(c, 2), w);
  rational(s, 1, 2);
  mul(at(c, 2), at(c, 2), s);
}

static void apply_small8(int r, struct fft_s *F, int s, Vector(x)) { 
  int    ir, is;
  Vector(c);
  Vector(x0); Vector(x1); Vector(x2); Vector(x3); Vector(x4); 
  Vector(x5); Vector(x6); Vector(x7);
  Scalar(t1); Scalar(t2); Scalar(t3); Scalar(t4); Scalar(t5); 
  Scalar(t6); Scalar(t7); Scalar(t8);
  Scalar(m0); Scalar(m1); Scalar(m2); Scalar(m3); Scalar(m4);
  Scalar(m5); Scalar(m6); Scalar(m7);
  Scalar(s1); Scalar(s2); Scalar(s3); Scalar(s4); 

  c  = F->priv.small.c;
  x0 = at(x, 0*s);  x1 = at(x, 1*s);
  x2 = at(x, 2*s);  x3 = at(x, 3*s);
  x4 = at(x, 4*s);  x5 = at(x, 5*s);
  x6 = at(x, 6*s);  x7 = at(x, 7*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      add( t1, x0, x4);
      add(t2, x2, x6);
      add(t3, x1, x5);
      sub(t4, x1, x5);
      add(t5, x3, x7);
      sub(t6, x3, x7);
      add(t7, t1, t2);
      add(t8, t3, t5);

      add(m0, t7, t8);
      sub(m1, t7, t8);
      sub(m2, t1, t2);
      sub(m3, x0, x4);
      sub(m4, t4, t6); mul(m4, m4, at(c, 0));
      sub(m5, t5, t3); mul(m5, m5, at(c, 1));
      sub(m6, x6, x2); mul(m6, m6, at(c, 1));
      add(m7, t4, t6); mul(m7, m7, at(c, 2));

      add(s1, m3, m4);
      sub(s2, m3, m4);
      add(s3, m6, m7);
      sub(s4, m6, m7);
      
      set(x0, m0);
      add(x1, s1, s3);
      add(x2, m2, m5);
      sub(x3, s2, s4);
      set(x4, m1);
      add(x5, s2, s4);
      sub(x6, m2, m5);
      sub(x7, s1, s3);

      x0 = at(x0, 1);  x1 = at(x1, 1);
      x2 = at(x2, 1);  x3 = at(x3, 1);
      x4 = at(x4, 1);  x5 = at(x5, 1);
      x6 = at(x6, 1);  x7 = at(x7, 1);
    }
    x0 = at(x0, -s + 8*s);  x1 = at(x1, -s + 8*s);
    x2 = at(x2, -s + 8*s);  x3 = at(x3, -s + 8*s);
    x4 = at(x4, -s + 8*s);  x5 = at(x5, -s + 8*s);
    x6 = at(x6, -s + 8*s);  x7 = at(x7, -s + 8*s);
  }
}

/* FFT on 9 points
   ---------------

   We implement the algorithm from Nussbaumer, 1982, sect. 5.5.7,
   modifying u -> -2 Pi/9 and restating the trigonometric constants
   as quantities in the 9-th roots of unity. In addition -1 = w(2)
   is needed. The algorithm is applicable iff 1/9, 1/2, and w(18)
   exist in the base field.

   3/2                                 -> c0
   -1/2                                -> c1
   (2 Cos[u] - Cos[2 u] - Cos[4 u])/3  -> c2
   (Cos[u] + Cos[2 u] - 2 Cos[4 u])/3  -> c3
   -(Cos[u] - 2 Cos[2 u] + Cos[4 u])/3 -> c4 
   -I Sin[3 u]                         -> c5
   I Sin[u]                            -> c6
   I Sin[4 u]                          -> c7
   I Sin[2 u]                          -> c8

   To avoid computations of the form x = -y - z we redefine 
   s0 -> -s0, s2 -> -s2, t15 -> -t15 which has implied a 
   flip of sign in c4. The constants are

   c0 = 3/2 
   c1 = -1/2
   c2 = w[9]/2 - w[9]^2/2 - w[9]^5/2
   c3 = -w[9]^4/2 - w[9]^5/2
   c4 = -w[9]/2 + w[9]^2/2 - w[9]^4/2
   c5 = 1/2 + w[9]^3
   c6 = -w[9]/2 - w[9]^2/2 - w[9]^5/2
   c7 = -w[9]^4/2 + w[9]^5/2
   c8 = -w[9]/2 - w[9]^2/2 - w[9]^4/2
*/

static void init_small9(Vector(c)) { 
  int    k, i;
  Scalar(s); 
  Scalar(w);
  static int coeffsTimes2[9][6] = {
    {  3,  0,  0, 0,  0,  0 },
    { -1,  0,  0, 0,  0,  0 }, 
    {  0,  1, -1, 0,  0, -1 }, 
    {  0,  0,  0, 0, -1, -1 }, 
    {  0, -1,  1, 0, -1,  0 },
    {  1,  0,  0, 2,  0,  0 }, 
    {  0, -1, -1, 0,  0, -1 }, 
    {  0,  0,  0, 0, -1,  1 }, 
    {  0, -1, -1, 0, -1,  0 }
  };
  
  rational(at(c, 0), 3, 2);
  rational(at(c, 1), -1, 2);
  for (k = 2; k < 9; ++k) {
    rational(at(c, k), coeffsTimes2[k][0], 2);
    for (i = 1; i < 6; ++i) {
      rational(s, coeffsTimes2[k][i], 2);
      rootOfUnity(w, 9, i);
      mul(s, s, w);
      add(at(c, k), at(c, k), s);
    }
  }
}

static void apply_small9(int r, struct fft_s *F, int s, Vector(x)) { 
  int ir, is;
  Vector(c);
  Vector(x0);  Vector(x1);  Vector(x2); 
  Vector(x3);  Vector(x4);  Vector(x5);
  Vector(x6);  Vector(x7);  Vector(x8);
  Scalar(t1);  Scalar(t2);  Scalar(t3);  Scalar(t4);  Scalar(t5); 
  Scalar(t6);  Scalar(t7);  Scalar(t8);  Scalar(t9);  Scalar(t10);
  Scalar(t11); Scalar(t12); Scalar(t13); Scalar(t14); Scalar(t15);
  Scalar(t16); 
  Scalar(m0);  Scalar(m1);  Scalar(m2);  Scalar(m3);  Scalar(m4); 
  Scalar(m5);  Scalar(m6);  Scalar(m7);  Scalar(m8);  Scalar(m9);
  Scalar(m10);
  Scalar(s0);  Scalar(s1);  Scalar(s2);  Scalar(s3);  Scalar(s4); 
  Scalar(s5);  Scalar(s6);  Scalar(s7);  Scalar(s8);  Scalar(s9);
  Scalar(s10); Scalar(s11); Scalar(s12);

  c  = F->priv.small.c;
  x0 = at(x, 0*s);  x1 = at(x, 1*s);  x2 = at(x, 2*s);
  x3 = at(x, 3*s);  x4 = at(x, 4*s);  x5 = at(x, 5*s);
  x6 = at(x, 6*s);  x7 = at(x, 7*s);  x8 = at(x, 8*s);
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      add(t1, x1, x8);
      add(t2, x2, x7);
      add(t3, x3, x6);
      add(t4, x4, x5);
      add(t5, t1, t2); add(t5, t5, t4);
      sub(t6, x1, x8);
      sub(t7, x7, x2);
      sub(t8, x3, x6);
      sub(t9, x4, x5);
      add(t10, t6, t7); add(t10, t10, t9);
      sub(t11, t1, t2);
      sub(t12, t2, t4);
      sub(t13, t7, t6);
      sub(t14, t7, t9);

      add(m0, x0, t3); add(m0, m0, t5);
      mul(m1, t3, at(c, 0)); 
      mul(m2, t5, at(c, 1)); 
      add(t15, t11, t12);
      mul(m3, t11, at(c, 2));
      mul(m4, t12, at(c, 3));
      mul(m5, t15, at(c, 4));
      add(s0, m3, m4);
      sub(s1, m5, m4);
      mul(m6, t10, at(c, 5));
      mul(m7, t8, at(c, 5));
      sub(t16, t14, t13);
      mul(m8, t13, at(c, 6));
      mul(m9, t14, at(c, 7));
      mul(m10, t16, at(c, 8));
      add(s2, m8, m9);
      sub(s3, m9, m10);
      add(s4, m0, m2); add(s4, s4, m2);
      sub(s5, s4, m1);
      add(s6, s4, m2);
      add(s7, s5, s0);
      add(s8, s1, s5);
      sub(s9, s5, s0); sub(s9, s9, s1);
      add(s10, m7, s2);
      sub(s11, m7, s3);
      sub(s12, m7, s2); add(s12, s12, s3);
      
      set(x0, m0);
      add(x1, s7, s10);
      sub(x2, s8, s11);
      add(x3, s6, m6);
      add(x4, s9, s12);
      sub(x5, s9, s12);
      sub(x6, s6, m6);
      add(x7, s8, s11);
      sub(x8, s7, s10);

      x0 = at(x0, 1);  x1 = at(x1, 1);  x2 = at(x2, 1);
      x3 = at(x3, 1);  x4 = at(x4, 1);  x5 = at(x5, 1);
      x6 = at(x6, 1);  x7 = at(x7, 1);  x8 = at(x8, 1);
    }
    x0 = at(x0, -s + 9*s);  x1 = at(x1, -s + 9*s);
    x2 = at(x2, -s + 9*s);  x3 = at(x3, -s + 9*s);
    x4 = at(x4, -s + 9*s);  x5 = at(x5, -s + 9*s);
    x6 = at(x6, -s + 9*s);  x7 = at(x7, -s + 9*s);
    x8 = at(x8, -s + 9*s);
  }
}

static void delete_small(struct fft_s *F) {
  Free(F->priv.small.c);
  delete_null(F);
}

fft_t *fft_new_small(int N) {
  fft_t *F;
  Vector(c);
  char  *msg;

  msg = fft_applicable(fft_small, N, 0, NULL);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }

  F            = fft_new_null(N);
  F->type      = fft_small;
  F->N         = N;
  F->deleteObj = delete_small;

  /* allocate space for precomputed constants */
  c =
    (fft_value *) Malloc(
       2 * fft_max_small * fft_valuesPerScalar * sizeof(fft_value)
    );
  F->priv.small.c = c;

  switch (N) {
    case 1: F->apply = apply_small1;                 break;
    case 2: F->apply = apply_small2;                 break;
    case 3: F->apply = apply_small3; init_small3(c); break;
    case 4: F->apply = apply_small4; init_small4(c); break;
    case 5: F->apply = apply_small5; init_small5(c); break;
    case 6: F->apply = apply_small6; init_small6(c); break;
    case 7: F->apply = apply_small7; init_small7(c); break;
    case 8: F->apply = apply_small8; init_small8(c); break;
    case 9: F->apply = apply_small9; init_small9(c); break;
    default:
      Error_int("unrecognized N = %d in fft_new_small()", N);
      delete_null(F);
      return NULL;
  }

  return F;
}

/* Rader-FFT
   =========

   The problematic part is how to compute the diagonal matrix
   for the cyclic convolution of length p-1. There has to be
   an accurate FFT available.
*/

static void print_perm(int n, fft_perm p[]) {
  int k;

  printf("{%d", p[0]);
  for (k = 1; k < n; k++)
    printf(", %d", p[k]);
  printf("}");
}

static void apply_rader(int r, struct fft_s *F, int s, Vector(x)) {
  int    N,                  /* length (prime) */
         ir, k, is;                 /* counter */
  Vector(x1);        /* pointer through signal */
  Vector(y1);        /* pointer through signal */
  Scalar(X0);        /* 0. component of result */
  Scalar(temp);                   /* temporary */
  Vector(y) = newa_vector(F->N - 1); /* buffer */

  N = F->N;
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      /* X0 = Sum(x[ir*s*N + k*s + is] : k in [0..N)) */
      x1 = at(x, ir*s*N + is);
      set(X0, x1);
      for (k = 1; k < N; ++k) {
        x1 = at(x1, s);
        add(X0, X0, x1);
      }

      /* [y[k] : k in [0..N-1)] = [x[ir*s*N + k*s + is] : k in [1..N)] */
      x1 = at(x, ir*s*N + 1*s + is);
      y1 = y;
      for (k = 0; k < N-1; ++k) {
        set(y1, x1);
        x1 = at(x1, s);
        y1 = at(y1, 1);
      }
  
      /* y = SR y */
      Apply_perm(1, N-1, F->priv.rader.SR, 1, y);

      /* y = DFT(N-1) y */
      fft_apply(1, F->priv.rader.fft_pMinus1, 1, y);

      /* y = diag(v_k : k) y */
      x1 = F->priv.rader.diag;
      y1 = y;
      for (k = 0; k < N-1; ++k) {
        mul(y1, y1, x1);
        x1 = at(x1, 1);
        y1 = at(y1, 1);
      }

      /* y = DFT(N-1) y */
      fft_apply(1, F->priv.rader.fft_pMinus1, 1, y);

      /* y = L y */
      Apply_perm(1, N-1, F->priv.rader.L, 1, y);

      /* y = y + x[ir*s*N + 0*s + is] * J_{N-1} */
      set(temp, at(x, ir*s*N + 0*s + is));
      y1 = y;
      for (k = 0; k < N-1; ++k) {
        add(y1, y1, temp);
        y1 = at(y1, 1);
      }

      /* [x[ir*s*N + k*s + is] : k in [1..N)] = [y[k] : k in [0..N-1)] */
      x1 = at(x, ir*s*N + 1*s + is);
      y1 = y;
      for (k = 0; k < N-1; ++k) {
        set(x1, y1);
        x1 = at(x1, s);
        y1 = at(y1, 1);
      }

      /* x[ir*s*N + 0*s +is] = X0 */
      set(at(x, ir*s*N + 0*s + is), X0);
    }
  }
  deletea_vector(y);
}

static void delete_rader(struct fft_s *F) {
  delete_vector(F->priv.rader.diag);
  fft_delete(F->priv.rader.fft_pMinus1);
  delete_perm(F->priv.rader.SR);
  delete_perm(F->priv.rader.L);
  delete_null(F);
}

fft_t *fft_new_rader(fft_t *fft_pMinus1) {
  fft_t    *F;
  int       p, q,   /* length, generator of (Z/pZ)^x */
            qk, q1,         /* q^k mod p, q^-1 mod p */
            k;                              /* index */
  Vector(   diag); /* the diagonal values for [0..p) */
  Vector(   w);           /* the p-th roots of unity */
  fft_perm *perm1, *perm2, *perm3;        /* buffers */
  char     *msg;

  p   = fft_pMinus1->N + 1;
  msg = fft_applicable(fft_rader, p, 0, NULL);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }
  perm1 = (fft_perm *) Malloc( (p-1) * sizeof(fft_perm) );
  perm2 = (fft_perm *) Malloc( (p-1) * sizeof(fft_perm) );
  perm3 = (fft_perm *) Malloc( (p-1) * sizeof(fft_perm) );

  /* create F */
  F            = fft_new_null(p);
  F->type      = fft_rader;
  F->N         = p;
  F->apply     = apply_rader;
  F->deleteObj = delete_rader;
  
  /* choose a small generator q for (Z/p Z)^x */
  q = generatorPrimeField(p);
  F->priv.rader.q = q;

  /* precompute output permutation L */
  q1 = powerMod(q, -1, p);
  qk = 1;
  for (k = 0; k < p-1; ++k) {
    perm1[k] = qk - 1; /* powerMod(q, (p-1)-k, p)-1 */
    qk       = mulMod(qk, q1, p);
  }
  for (k = 0; k < p-1; ++k)
    perm2[perm1[k]] = k;
  F->priv.rader.L = new_perm(p-1, perm2);

  /* perm1 = R */
  qk = 1;
  for (k = 0; k < p-1; ++k) {
    perm1[k] = qk - 1; /* powerMod(q, k, p)-1 */
    qk       = mulMod(qk, q, p);
  }

  /* perm2 = S (reversing perm from the inverse FFT) */
  for (k = 0; k < p-1; ++k)
    perm2[k] = k;
  for (k = 1; k <= ((p-1)-1)/2; ++k) {
    perm2[k]       = (p-1)-k;
    perm2[(p-1)-k] = k;
  }

  /* perm3 = perm2 * perm1 */
  for (k = 0; k < p-1; ++k)
    perm3[k] = perm1[perm2[k]];

  /* precompute reversing perm * input perm */
  F->priv.rader.SR = new_perm(p-1, perm3);

  /* DFT(p-1) */
  F->priv.rader.fft_pMinus1 = 
    (struct fft_s *) fft_pMinus1;

  /* precompute the diagonal values 
       diag[k] = w[p]^powerMod(q, k, p)

     (We reuse the values in perm1[k] = powerMod(q, k, p)-1.)
  */

  /* w = p-th roots of unity */
  w = new_vector(p);
  rootsOfUnity(w, p);

  /* diag = [ w(p)^(q^k) : k in [0..p-1) ] */
  diag = new_vector(p-1);
  for (k = 0; k < p-1; ++k)
    set(at(diag, k), at(w, perm1[k]+1));

  /* diag = 1/(p-1) * diag (from DFT(p-1)^-1) */
  for (k = 0; k < p-1; ++k)
    mul(at(diag, k), at(diag, k), fft_pMinus1->inverseN);

  /* diag = DFT(p-1) diag */
  fft_apply(1, fft_pMinus1, 1, diag);
  F->priv.rader.diag = diag;

  delete_vector(w);
  Free(perm1);
  Free(perm2);
  Free(perm3);
  return F;
}

/* Good-Thomas-FFT
   ===============
*/

static void apply_goodThomas(int r, struct fft_s *F, int s, Vector(x)) {
  int N, nq, *q,               /* fetched from F */
      R, S,   /* q[0] .. q[a-1], q[a] .. q[nq-1] */
      a;                                /* index */

  N  = F->N;
  nq = F->priv.goodThomas.nq;
  q  = F->priv.goodThomas.q;

  /* x = (1_r (x) R (x) (1_s)) . x */
  Apply_perm(r, N, F->priv.goodThomas.R, s, x);

  /* apply tensored FFT's */
  R = r * 1;
  S = N * s;
  for (a = 0; a < nq; ++a) {
    S = S / q[a];
    fft_apply(R, F->priv.goodThomas.fft_q[a], S, x);
    R = R * q[a];
  }

  /* x = (1_r (x) L (x) (1_s)) . x */
  Apply_perm(r, N, F->priv.goodThomas.L, s, x);
}

static void delete_goodThomas(struct fft_s *F) {
  int a;

  for (a = F->priv.goodThomas.nq-1; a >= 0; --a)
    fft_delete(F->priv.goodThomas.fft_q[a]);

  delete_perm(F->priv.goodThomas.R);
  delete_perm(F->priv.goodThomas.L);
  delete_null(F);
}

fft_t *fft_new_goodThomas(int nq, fft_t *fft_q[]) {
  fft_t    *F;
  int       q[fft_max_factors],       /* the lengths of fft_q[] */
            N,                                 /* length of FFT */
            k, k1[fft_max_factors],     /* index k, digits of k */
            Lk, Rk,                        /* images L(k), R(k) */
            a;                         /* counter for [0..nq-1] */
  fft_perm *perm;               /* intermediate buffer for L, R */
  char     *msg;

  N = 1;
  for (a = 0; a < nq; ++a) {
    q[a] = fft_q[a]->N;
    N    = N * q[a];
  }
  msg = fft_applicable(fft_goodThomas, N, nq, q);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }
  perm = (fft_perm *) Malloc( N * sizeof(fft_perm) );

  /* create F */
  F            = fft_new_null(N);
  F->type      = fft_goodThomas;
  F->N         = N;
  F->apply     = apply_goodThomas;
  F->deleteObj = delete_goodThomas;

  /* copy the vector q into F */
  F->priv.goodThomas.nq = nq;
  for (a = 0; a < nq; ++a) 
    F->priv.goodThomas.q[a] = q[a];

  /* precompute permutation L */
  for (k = 0; k < N; ++k) {

    /* compute image L(k) */
    Lk = mod(k, q[0]);
    for (a = 1; a < nq; ++a)
      Lk = Lk*q[a] + mod(k, q[a]);

    /* store image */
    perm[k] = Lk;
  }
  F->priv.goodThomas.L = new_perm(N, perm);

  /* precompute permutation R */
  for (a = 0; a < nq; ++a)
    k1[a] = 0;
  for (k = 0; k < N; ++k) {

    /* compute image R(k) */
    Rk = 0;
    for (a = 0; a < nq; ++a)
      Rk = addMod(Rk, mulMod(k1[a], N/q[a], N), N);

    /* store image */
    perm[k] = Rk;

    /* increment digits (k1[0], .., k1[nq-1]) of k */
    a = nq-1;
    k1[a]++;
    while ((a > 0) && (k1[a] == q[a])) {
      k1[a] = 0;
      --a;
      k1[a]++;
    }
  }
  F->priv.goodThomas.R = new_perm(N, perm);

  /* store the smaller FFT's */
  for (a = 0; a < nq; ++a)
    F->priv.goodThomas.fft_q[a] = (struct fft_s *) fft_q[a];

  Free(perm);
  return F;
}

/* Cooley-Tukey-FFT
   ================
*/

static void apply_cooleyTukey(int r, struct fft_s *F, int s, Vector(x)) {
  int    N, n, m,  /* length, N = n m */
         ir, k, is;        /* counter */
  Vector(x1);    /* pointer through x */
  Scalar(w);    /* N-th root of unity */

  N = F->N;
  n = F->priv.cooleyTukey.n;
  m = F->priv.cooleyTukey.m;

  /* x = (1_r (x) DFT(n) (x) 1_{m s}) x */
  fft_apply(r, F->priv.cooleyTukey.fft_n, m*s, x);

  /* x = (1_r (x) diag(w[N]^t(k) : k) (x) 1_s) x */
  for (k = 0; k < N; ++k) {
    set(w, 
      at(
        F->priv.cooleyTukey.rootOfUnity, 
        mod(k,m)*div(k,m)
      )
    );
    x1 = at(x, k*s);
    for (ir = 0; ir < r; ++ir) {
      for (is = 0; is < s; ++is) {

        /* x[ir*N*s + k*s + is] *= w */
        mul(x1, x1, w);
        x1 = at(x1, 1);
      }
      x1 = at(x1, -s + N*s);
    }
  }

  /* x = (1_{r*n} (x) DFT(m) (x) 1_s) x */
  fft_apply(r*n, F->priv.cooleyTukey.fft_m, s, x);

  /* x = (1_r (x) L (x) 1_s) x */
  Apply_perm(r, N, F->priv.cooleyTukey.L, s, x);
}

static void delete_cooleyTukey(struct fft_s *F) {
  fft_delete(F->priv.cooleyTukey.fft_m);
  fft_delete(F->priv.cooleyTukey.fft_n);
  delete_perm(F->priv.cooleyTukey.L);
  delete_vector(F->priv.cooleyTukey.rootOfUnity);
  delete_null(F);
}

fft_t *fft_new_cooleyTukey(fft_t *fft_n, fft_t *fft_m) {
  fft_t    *F;
  int       N, n, m,               /* length N = n m */
            k;                       /* index [0..N) */
  fft_perm *perm;       /* intermediate buffer for L */
  char     *msg;
  int       q[2];

  n = fft_n->N; q[0] = n;
  m = fft_m->N; q[1] = m;
  N = n * m;
  msg = fft_applicable(fft_cooleyTukey, N, 2, q);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }
  perm = (fft_perm *) Malloc(N * sizeof(fft_perm));

  /* create F */
  F                     = fft_new_null(N);
  F->type               = fft_cooleyTukey;
  F->N                  = N;
  F->apply              = apply_cooleyTukey;
  F->deleteObj          = delete_cooleyTukey;
  F->priv.cooleyTukey.n = n;
  F->priv.cooleyTukey.m = m;

  /* precompute the N-th roots of unity */
  F->priv.cooleyTukey.rootOfUnity = new_vector(N);
  rootsOfUnity(F->priv.cooleyTukey.rootOfUnity, N);

  /* precompute output-permutation L */
  for (k = 0; k < N; ++k)
    perm[k] = mod(k, n)*m + div(k, n);
  F->priv.cooleyTukey.L = new_perm(N, perm);

  /* construct DFT(n), DFT(m) */
  F->priv.cooleyTukey.fft_n = (struct fft_s *) fft_n;  
  F->priv.cooleyTukey.fft_m = (struct fft_s *) fft_m;

  Free(perm);
  return F;
}

/* q-Power-FFT
   ===========

   The interesting part is the multiplication with the twiddle
   factors. In particular, for a in [1..e-1] we need to compute

      x = (1_r (x) diag(w^t(a, k) : k in [0..q)) (x) 1_s) . x 

   where 

      t(a, k) = (k % q^(a+1))/q^a * (k % q^a) * q^((e-1)-a).

   Instead of computing this directly, we turn it the other way round:
   Fix T and enumerate all k for which t(a, k) = T. Since w^0 = 1,
   we can ommit T = 0, so the program to enumerate the pairs (T, k) is

     for T1 in [1..(q-1)(q^a - 1)]; T = T1 q^((e-1) - a)
       enumerate k such that
         T1 = ((k % q^(a+1)) % q^a)*((k % q^(a+1)) / q^a)

   transforms by k1 = (k % q^(a+1)) / q^a to

     for T1 in [1..(q-1)(q^a - 1)]; T = T1 q^((e-1) - a)
       for k1 in [1..q) such that k1 divides T1 and T1/k1 < q^a
         enumerate k such that
           T1/k1 = (k % q^(a+1)) % q^a and 
           k1    = (k % q^(a+1)) / q^a

   transforms by composing div(-,q^a) q^a + mod(-,q^a) to

     for T1 in [1..(q-1)(q^a - 1)]; T = T1 q^((e-1) - a)
       for k1 in [1..q) such that k1 divides T1 and T1/k1 < q^a
         enumerate k s.t. k1 q^a + T1/k1 = k % q^(a+1)

   transforms by T1/k1 < q^a <==> k1 >= div(T1, q^a)+1 to

     for T1 in [1..(q-1)(q^a - 1)]; T = T1 q^((e-1) - a)
       for k1 in [div(T1,q^a)+1..q) such that k1 divides T1
         for k2 in [0..q^((e-1)-a))
           k = k2 q^(a+1) + k1 q^a + T1/k1.

   The result is

[Mathematica-begin]

(* q-Power-FFT Twiddle-Exponents, SE, 18.6.96, Mathematica v2.0 *)

tEnumerated[q_, e_, a_] := 
  Module[{tVector, T, k},
    tVector = Table[0, {q^e}];
    Do[
      Do[
        If[Mod[T1, k1] == 0,
          Do[
            T = q^((e-1) - a) * T1;
            k = k2 q^(a+1) + k1 q^a + T1/k1;

            tVector[[1 + k]] = T

          , {k2, 0, q^((e-1)-a) - 1}
          ]
        ]
      , {k1, Quotient[T1, q^a]+1, q - 1}
      ]
    , {T1, 1, (q-1)*(q^a - 1)}
    ];
    tVector
  ]

(* to check *)
t[q_, e_, a_]     := Table[t[q, e, a, k], {k, 0, q^e - 1}]
t[q_, e_, a_, k_] :=
  Quotient[Mod[k, q^(a+1)], q^a] * Mod[k, q^a] * q^((e-1) - a)

[Mathematica-end]
*/

static void apply_2Power(int r, struct fft_s *F, int s, Vector(x)) {
  int    N, e,                /* fetched from F */
         a, qa, Qa;         /* q^a, q^((e-1)-a) */
  Vector(xk);                 /* signal pointer */
  int    dxk,               /* inner step of xk */
         T1,  /* twiddle exponent divided by Qa */
         k2,                       /* temporary */
         ir, is;            /* counter for r, s */
  Vector(w);                    /* w[N]^(T1 Qa) */

  N = F->N;
  e = F->priv.qPower.e;

  /* x = (1_r (x) R (x) (1_s)) . x */
  Apply_perm(r, N, F->priv.qPower.R, s, x);

  /* x = (1_{r q^(e-1)} (x) DFT_q (x) 1_{q^0 s}) . x */
  apply_small2(r * (N/2), F->priv.qPower.fft_q, 1 * s, x);

  /* apply twiddles and tensored DFT_q; for convenience:
       qa = q^a
       Qa = q^((e-1) - a)  for all a in [1..e-1]
  */
  qa = 1;
  Qa = N/2;
  for (a = 1; a <= e-1; ++a) {
    qa *= 2;
    Qa /= 2;

    /* x = (1_r (x) diag(w^t(a, k) : k in [0..N)) (x) 1_s) . x
       where t(a, k) = (k % (q*qa))/qa * (k % qa) * Qa.
       (See comment above.)
    */
    w   = at(F->priv.qPower.rootOfUnity, Qa);
    dxk = -s + 2*qa*s;
    for (T1 = 1; T1 < qa; ++T1) {
      xk = at(x, (qa + T1)*s);
      for (ir = 0; ir < r; ++ir) {
        for (k2 = 0; k2 < Qa; ++k2) {
          for (is = 0; is < s; ++is) {
            mul(xk, xk, w);
            xk = at(xk, 1);
          }
          xk = at(xk, dxk);
        }
      }
      w = at(w, Qa);
    }

    /* x = (1_{r q^((e-1)-a)} (x) DFT_q (x) 1_{q^a s}) . x */
    apply_small2(r * Qa, F->priv.qPower.fft_q, qa * s, x);
  }
}

static void apply_qPower(int r, struct fft_s *F, int s, Vector(x)) {
  int    N, q, e,            /* fetched from F */
         a, qa, Qa;        /* q^a, q^((e-1)-a) */
  Vector(xk);                /* signal pointer */
  Scalar(w);             /* N-th root of unity */
  fft_t *fft_q;         /* the FFT of length q */

  /* treat the 2-power case individually */
  if (F->priv.qPower.q == 2) {
    apply_2Power(r, F, s, x);
    return;
  }

  /* fetch some constants */
  N     = F->N;
  q     = F->priv.qPower.q;
  e     = F->priv.qPower.e;
  fft_q = F->priv.qPower.fft_q;

  /* x = (1_r (x) R (x) (1_s)) . x */
  Apply_perm(r, N, F->priv.qPower.R, s, x);

  /* x = (1_{r q^(e-1)} (x) DFT_q (x) 1_{q^0 s}) . x */
  fft_apply(r * (N/q), fft_q, 1 * s,  x);

  /* apply twiddles and tensored DFT_q; for convenience:
       qa = q^a
       Qa = q^((e-1) - a)  for all a in [1..e-1]
  */
  qa = 1;
  Qa = N/q;
  for (a = 1; a <= e-1; ++a) {
    qa *= q; 
    Qa /= q;
 
    /* x = (1_r (x) diag(w^t(a, k) : k in [0..N)) (x) 1_s) . x
       where t(a, k) = (k % (q*qa))/qa * (k % qa) * Qa.
       (See comment above.)
    */

#if defined(SimpleTwiddle)

    { int k, t, ir, is;

      for (k = 0; k < N; ++k) {
        t = (k % (q*qa))/qa * (k % qa) * Qa;
        if (t > 0) {
          set(w, at(F->priv.qPower.rootOfUnity, t));
          xk = at(x, k*s);
          for (ir = 0; ir < r; ++ir) {
            for (is = 0; is < s; ++is) {
              mul(xk, xk, w);
              xk = at(xk, 1);
            }
            xk = at(xk, -s + N*s);
          }
        }
      }
    }

#else /* UnrolledTwiddle */

    { int dxk = -s + q*qa*s, /* inner step of xk */
          T1,  /* twiddle exponent divided by Qa */
          k1, k2,                   /* temporary */
          ir, is;            /* counter for r, s */

      for (T1 = 1; T1 <= (q-1)*(qa-1); ++T1) {

    /* w = w[N]^(T1 Qa) */
    set(w, at(F->priv.qPower.rootOfUnity, T1*Qa));

    for (k1 = (T1/qa)+1; k1 < q; ++k1) {
      if (T1 % k1 == 0) {
        xk = at(x, (k1*qa + T1/k1)*s);
        for (ir = 0; ir < r; ++ir) {
          for (k2 = 0; k2 < Qa; ++k2) {
        for (is = 0; is < s; ++is) {

          /* x[ir*N*s + k*s + is] *= w
             where k = k2*q*qa + k1*qa + T1/k1
          */
          mul(xk, xk, w);
          xk = at(xk, 1);
        }
        xk = at(xk, dxk);
          }
       /* xk = at(xk, -Qa*q*qa*s + N*s); (Note: q*qa*Qa = N) */ 
        }
      }
    }
      }
    }

#endif

    /* x = (1_{r q^((e-1)-a)} (x) DFT_q (x) 1_{q^a s}) . x */
    fft_apply(r * Qa, fft_q, qa * s,  x);
  }
}

static void delete_qPower(struct fft_s *F) {
  fft_delete(F->priv.qPower.fft_q);
  delete_perm(F->priv.qPower.R);
  delete_vector(F->priv.qPower.rootOfUnity);
  delete_null(F);
}

fft_t *fft_new_qPower(fft_t *fft_q, int e) {
  fft_t    *F;
  int       N, q,                        /* q^e */
            k, Rk,               /* index, R(k) */ 
            k1, a;        /* temporary, counter */
  fft_perm *perm;               /* buffer for R */
  char     *msg;
  int       qvec[2];

  q       = fft_q->N;
  qvec[0] = q;
  qvec[1] = e;
  N    = 1;
  for (k = 0; k < e; ++k)
    N = N * qvec[0];
  msg = fft_applicable(fft_qPower, N, 2, qvec);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }
  perm = (fft_perm *) Malloc( N * sizeof(fft_perm) );

  /* create F */
  F                = fft_new_null(N);
  F->type          = fft_qPower;
  F->N             = N;
  F->apply         = apply_qPower;
  F->deleteObj     = delete_qPower;
  F->priv.qPower.q = q;
  F->priv.qPower.e = e;

  /* precompute the N-th roots of unity */
  F->priv.qPower.rootOfUnity = new_vector(N);
  rootsOfUnity(F->priv.qPower.rootOfUnity, N);

  /* precompute the permutation R */
  for (k = 0; k < N; ++k)
    perm[k] = k;
  for (k = 0; k < N; ++k) {

    /* Rk = q-digits of k reversed */
    Rk = mod(k, q);
    k1 = div(k, q);
    for (a = 1; a < e; ++a) {
      Rk = Rk*q + mod(k1, q);
      k1 = k1 / q;
    }

    /* swap k and Rk (but only once) */
    if (Rk < k) {
      perm[k]  = Rk;
      perm[Rk] = k;
    }
  }
  F->priv.qPower.R = new_perm(N, perm);

  /* construct the FFT on q points */
  F->priv.qPower.fft_q = (struct fft_s *) fft_q;

  Free(perm);
  return F;
}

/* Bluestein-FFT
   =============
*/

static void apply_bluestein(int r, struct fft_s *F, int s, Vector(x)) {
  int    N, M,                /* length, embedding length */
         ir, k, is;                /* counter for r, N, s */
  Vector(x1);                 /* pointer to signal values */
  Vector(y1);                 /* pointer to signal values */
  Scalar(w);                  /* a (2 N)-th root of unity */
  Scalar(temp);                              /* temporary */
  Vector(y) = newa_vector(F->priv.bluestein.M); /* buffer */

  N = F->N;
  M = F->priv.bluestein.M;

  /* x = (1_r (x) diag(w(2 N)^(k^2) : k) (x) 1_s) x */
  x1 = x;
  for (ir = 0; ir < r; ++ir) {
    for (k = 0; k < N; ++k) {
      set(w, at(F->priv.bluestein.diagN, k));
      for (is = 0; is < s; ++is) {
        mul(x1, x1, w);
        x1 = at(x1, 1);
      }
      x1 = x1;
    }
  }

  /* cyclic convolution of length M >= 2*N */
  for (ir = 0; ir < r; ++ir) {
    for (is = 0; is < s; ++is) {

      /* buffer = (ir, is)-component of x */
      x1 = at(x, ir*N*s + is);
      y1 = y;
      for (k = 0; k < N; ++k) {
        set(y1, x1);
        x1 = at(x1, s);
        y1 = at(y1, 1);
      }
      for (k = N; k < M; ++k) {
        rational(y1, 0, 1);
        y1 = at(y1, 1);
      }

      /* buffer = S buffer for reversing perm S */
      for (k = 1; k <= (M-1)/2; ++k) {
        set(temp,         at(y, k)); 
        set(at(y, k),     at(y, M - k));
        set(at(y, M - k), temp);
      }

      /* buffer = DFT(M) buffer */
      fft_apply(1, F->priv.bluestein.fft_M, 1, y);
    
      /* buffer = diagM . buffer */
      x1 = F->priv.bluestein.diagM;
      y1 = y;
      for (k = 0; k < M; ++k) {
        mul(y1, y1, x1);
        x1 = at(x1, 1);
        y1 = at(y1, 1);
      }

      /* buffer = DFT(M) buffer */
      fft_apply(1, F->priv.bluestein.fft_M, 1, y);
    
      /* x = first N components of buffer */
      x1 = at(x, ir*N*s + is);
      y1 = y;
      for (k = 0; k < N; ++k) {
        set(x1, y1);
        x1 = at(x1, s);
        y1 = at(y1, 1);
      }
    }
  }

  /* x = (1_r (x) diag(w[2 N]^(k^2) : k) (x) 1_s) x */
  x1 = x;
  for (ir = 0; ir < r; ++ir) {
    for (k = 0; k < N; ++k) {
      set(w, at(F->priv.bluestein.diagN, k));
      for (is = 0; is < s; ++is) {
        mul(x1, x1, w);
        x1 = at(x1, 1);
      }
    }
  }
  deletea_vector(y);
}

static void delete_bluestein(struct fft_s *F) {
  delete_vector(F->priv.bluestein.diagM);
  fft_delete(F->priv.bluestein.fft_M);
  delete_vector(F->priv.bluestein.diagN);
  delete_null(F);
}

fft_t *fft_new_bluestein(int N, fft_t *fft_M) {
  fft_t *F;
  int    M,              /* length to embed in */
         k,                           /* index */
         kSqr;      /* k^2 mod 2 N or the like */
  Vector(diagM);    /* the diag(-) of length M */
  Vector(w);    /* the (2 N)-th roots of unity */
  char  *msg;
  int    q[1];

  M    = fft_M->N;
  q[0] = M;
  msg  = fft_applicable(fft_bluestein, N, 1, q);
  if (msg != NULL) {
    Error(msg);
    return NULL;
  }

  /* create F */
  F                   = fft_new_null(N);
  F->type             = fft_bluestein;
  F->N                = N;
  F->apply            = apply_bluestein;
  F->deleteObj        = delete_bluestein;
  F->priv.bluestein.M = M;

  /* w = (2 N)-th roots of unity */
  w = new_vector(2*N);
  rootsOfUnity(w, 2*N);

  /* diagN = [ w(2 N)^(k^2) : k in [0..N) ] */
  F->priv.bluestein.diagN = new_vector(N);
  kSqr = 0;
  for (k = 0; k < N; ++k) {
    set(at(F->priv.bluestein.diagN, k), at(w, kSqr)); 
    kSqr = addMod(kSqr, 2*k + 1, 2*N);
  }

  /* construct the FFT on M points */
  F->priv.bluestein.fft_M = (struct fft_s *) fft_M;

  /* diagM = 1/M * DFT(M) * [ v_k : k in [0..M) ] where
     v_k = w(2 N)^(-k^2)               for 0   <= k < N
         = 0                           for N   <= k < M-N
         = w(2 N)^(-(k-(M-N))^2 + N^2) for M-N <= k < M

     (Note kSqr = -k^2 mod 2 N for all k in the loop)
  */
  diagM = new_vector(M);
  kSqr  = 0;
  for (k = 0; k < N; ++k) {
    set(at(diagM, k), at(w, kSqr)); 
    kSqr = addMod(kSqr, -2*k - 1, 2*N);
  }
  for (k = N; k < M-N; ++k)
    rational(at(diagM, k), 0, 1);
  for (k = M-N; k < M; ++k)
    set(at(diagM, k), at(diagM, k-(M-N)));
  if (mod(N, 2) != 0)
    for (k = M-N; k < M; ++k)
      mul(at(diagM, k), at(diagM, k), at(w, N));
  for (k = 0; k < M; ++k)
    mul(at(diagM, k), at(diagM, k), fft_M->inverseN);

  /* diagM = DFT(M) diagM */
  fft_apply(1, fft_M, 1, diagM);
  F->priv.bluestein.diagM = diagM;

  delete_vector(w);
  return F;
}

/* Composition Strategies for FFT-methods
   ======================================
*/

fft_t *fft_new_simple(int N) {
  int   q, e, M, N1;
  char *msg;
  int   u[fft_max_factors];

  if (!( (1 <= N) && (N <= fft_max_length) ))
    Error_int2(
      "N = %d must be in [1..%d] in fft_new_simple()",
           N,                fft_max_length
    );

  if (N <= fft_max_small) {
    msg = fft_applicable(fft_small, N, 0, NULL);
    if (msg == NULL)
      return fft_new_small(N);
    Free(msg);
  }

  /* try N = q^e for 1 < q <= fft_max_small */
  for (q = fft_max_small; q > 1; --q) {
    N1 = N;
    e  = 0;
    while (mod(N1, q) == 0) {
      N1 /= q;
      ++e;
    }
    if (N1 == 1) {
      u[0] = q;
      u[1] = e;
      msg  = fft_applicable(fft_qPower, N, 2, u);
      if (msg == NULL)
        return fft_new_qPower(fft_new_small(q), e);
      Free(msg);
    }
  }
  
  /* use Bluestein with M = q^e >= 2*N and q <= fft_max_small */
  M = 2*N;
  while (M <= 4*N) {
    for (q = fft_max_small; q > 1; --q) {
      N1 = M;
      e  = 0;
      while (mod(N1, q) == 0) {
        N1 /= q;
        ++e;
      }
      if (N1 == 1) {
        u[0] = M;
        msg  = fft_applicable(fft_bluestein, N, 1, u);
        if (msg == NULL)
          return 
            fft_new_bluestein(
              N,
              fft_new_qPower(fft_new_small(q), e)
            );
        Free(msg);
      }
    }
    ++M;
  }

  /* M became too large */
  return fft_new_direct(N);
}

fft_t *fft_new_prettyGood(int N) {
  int    np, p[fft_max_factors], e[fft_max_factors], /* factors */
         u[fft_max_factors],         /* args for fft_applicable */
         q, e1;                                   /* parameters */
  char  *msg;

  if (!( (1 <= N) && (N <= fft_max_length) ))
    Error_int2(
      "N = %d must be in [1..%d] in fft_new_simple()",
           N,                fft_max_length
    );

  /* look for an exception */
  {
     int i, M;
 
#define T fft_bluesteinExceptionTable

     i = 0;
     while ((T[i][0] != -1) && (T[i][0] < N))
       ++i;
     if (T[i][0] == N) {

       /* N is an exception, use bluestein[N, <fft(M)>] */
       M    = T[i][1];
       u[0] = M;
       msg  = fft_applicable(fft_bluestein, N, 1, u);
       if (msg == NULL)
         return 
           fft_new_bluestein(
             N, 
             fft_new_prettyGood(M)
           );
       Free(msg);
     }

#undef T
  }

  /* try small */
  if (N <= fft_max_small) {
    msg = fft_applicable(fft_small, N, 0, NULL);
    if (msg == NULL)
      return fft_new_small(N);
    Free(msg);
  }

  /* try N = 2^e */
  { int N1;

    N1 = N;
    e1 = 0;
    while (N1 % 2 == 0) {
      N1 /= 2;
      ++e1;
    }
    if (N1 == 1) {
      u[0] = 2;
      u[1] = e1;
      msg  = fft_applicable(fft_qPower, N, 2, u);
      if (msg == NULL)
        return fft_new_qPower(fft_new_small(2), e1);
      Free(msg);
    }
  }

  /* try N = q^e for 2 < q <= fft_max_small */
  for (q = fft_max_small; q > 2; --q) {
    int N1;

    N1 = N;
    e1 = 0;
    while (mod(N1, q) == 0) {
      N1 /= q;
      ++e1;
    }
    if (N1 == 1) {
      u[0] = q;
      u[1] = e1; 
      msg  = fft_applicable(fft_qPower, N, 2, u);
      if (msg == NULL)
        return fft_new_qPower(fft_new_small(q), e1);
      Free(msg);
    }
  }

  fft_factor(N, &np, p, e);
  if (np > 1) {
    fft_t *fft_q[fft_max_factors];
    int    nq, q[fft_max_factors],
           iq, jq,
           temp;

    /* q[iq] = p[iq]^e[iq] */
    nq = np;
    for (iq = 0; iq < nq; ++iq)
      q[iq] = powerMod(p[iq], e[iq], N);

    /* sort q[] */
    for (iq = 0; iq < nq-1; ++iq)
      for (jq = iq+1; jq < nq; ++jq)
        if (q[jq] < q[iq]) {
          temp  = q[iq];
          q[iq] = q[jq];
          q[jq] = temp;
        }

    /* create the FFT's for q[] */
    for (iq = 0; iq < nq; ++iq)
      fft_q[iq] = fft_new_prettyGood(q[iq]);

    return fft_new_goodThomas(nq, fft_q);
  }

  if ((np == 1) && (e[0] == 1)) {
    msg = fft_applicable(fft_rader, p[0], 0, NULL);
    if (msg == NULL)
      return fft_new_rader( fft_new_prettyGood(p[0]-1) );
    Free(msg);
  }

  if ((np == 1) && (e[0] > 1)) {
    return fft_new_qPower( fft_new_prettyGood(p[0]), e[0]);
  }

  return fft_new_direct(N);
}

/* Print FFT-Method as a Tree
   ==========================
*/

static int print1(char *out, int indent, fft_t *F) {
  int n, /* length of string produced (excl. trailing 0) */
      dn,                     /* a new portition of text */
      a;                              /* counter for q[] */

#define print_string(s) \
  { dn = sprintf(out+n, s); \
    if (dn < 0) Error("sprintf() failed"); n += dn; }

#define print_string_int(s, x) \
  { dn = sprintf(out+n, s, x); \
    if (dn < 0) Error("sprintf() failed"); n += dn; }

#define print_fft(indent, F) \
  { dn = print1(out+n, indent, F); \
    if (dn < 0) Error("sprintf() failed"); n += dn; }

#define print_newline(indent) \
  { int i; print_string("\n"); \
    for (i = 0; i < indent; ++i) print_string("  "); }

  n = 0;
  switch (F->type) {

    case fft_null:
      print_string_int("null[%d]", F->N);
      break;

    case fft_direct:
      print_string_int("direct[%d]", F->N);
      break;

    case fft_small:
      print_string_int("small[%d]", F->N);
      break;

    case fft_rader:
      if (indent < 0) {
        print_string("rader[");
        print_fft(indent, F->priv.rader.fft_pMinus1);
        print_string("]");
      } else {
        print_string_int("rader[ (* N = %d (prime) *)", F->N);
        print_newline(indent+1);
        print_fft(indent+1, F->priv.rader.fft_pMinus1);
        print_newline(indent);
        print_string("]");
      }
      break;

    case fft_goodThomas:
      if (indent < 0) {
        print_string("goodThomas[");
        print_fft(indent, F->priv.goodThomas.fft_q[0]);
        for (a = 1; a < F->priv.goodThomas.nq; ++a) {
          print_string(",");
          print_fft(indent, F->priv.goodThomas.fft_q[a]);
        }
        print_string("]");
      } else {
        print_string_int("goodThomas[ (* N = %d = ", F->N);
        print_string_int("%d", F->priv.goodThomas.q[0]);
        for (a = 1; a < F->priv.goodThomas.nq; ++a)
          print_string_int(" * %d", F->priv.goodThomas.q[a]);
        print_string(" *)");
        print_newline(indent+1);
        print_fft(indent+1, F->priv.goodThomas.fft_q[0]);
        for (a = 1; a < F->priv.goodThomas.nq; ++a) {
          print_string(",");
          print_newline(indent+1);
          print_fft(indent+1, F->priv.goodThomas.fft_q[a]);
        }
        print_newline(indent);
        print_string("]");
      }
      break;

    case fft_cooleyTukey:
      if (indent < 0) {
        print_string("cooleyTukey[");
        print_fft(indent, F->priv.cooleyTukey.fft_n);
        print_string(",");
        print_fft(indent, F->priv.cooleyTukey.fft_m);
        print_string("]");
      } else {
        print_string_int("cooleyTukey[ (* N = %d = ", F->N);
        print_string_int("%d * ", F->priv.cooleyTukey.n);
        print_string_int("%d *)", F->priv.cooleyTukey.m);
        print_newline(indent+1);
        print_fft(indent+1, F->priv.cooleyTukey.fft_n);
        print_string(",");
        print_newline(indent+1);
        print_fft(indent+1, F->priv.cooleyTukey.fft_m);
        print_newline(indent);
        print_string("]");
      }
      break;

    case fft_qPower:
      if (indent < 0) {
        print_string("qPower[");
        print_fft(indent, F->priv.qPower.fft_q);
        print_string_int(",%d]", F->priv.qPower.e);
      } else {
        print_string_int("qPower[ (* N = %d = ", F->N);
        print_string_int("%d^", F->priv.qPower.q);
        print_string_int("%d *)", F->priv.qPower.e);
        print_newline(indent+1);
        print_fft(indent+1, F->priv.qPower.fft_q);
        print_string(",");
        print_newline(indent+1);
        print_string_int("%d", F->priv.qPower.e);
        print_newline(indent);
        print_string("]");
      }
      break;

    case fft_bluestein:
      if (indent < 0) {
        print_string_int("bluestein[%d,", F->N);
        print_fft(indent, F->priv.bluestein.fft_M);
        print_string("]");
      } else {
        print_string_int("bluestein[ %d, (* = N ", F->N);
        print_string_int("<= %d/2 *)", F->priv.bluestein.M);
        print_newline(indent+1);
        print_fft(indent+1, F->priv.bluestein.fft_M);
        print_newline(indent);
        print_string("]");
      }
      break;

    default:
      print_string_int("unrecognized[{type -> %d, ", F->type);
      print_string_int("N -> %d}]", F->N);
      break;
  }
  return n;

#undef print_string
#undef print_string_int
#undef print_fft
#undef print_newline
}

char *fft_print(int indent, fft_t *F) {
  char *out, *out1;
  int   n;

  /* print into sufficiently large buffer first, then copy */
  out1 = (char *) Malloc( 10000 * sizeof(char) );
  n    = print1(out1, indent, F);
  out  = (char *) Malloc( (n+1) * sizeof(char) );
  strcpy(out, out1);
  Free(out1);
  return out;
}

/* Parse FFT-Method from a String
   ==============================
*/

/* (int) parse_int(msg, in, x)
     parses an integer from the string in and stores it into x.
     The function returns the number of characters consumed and
     stores an error message if it has not been successful
*/

static int parse_int(char **msg, char *in, int *x) {
  int n, /* length of the integer literal */ 
      i;              /* counter for in[] */

  /* the integer is in[0], .., in[n-1] */
  n = 0;
  while (('0' <= in[n]) && (in[n] <= '9'))
    ++n;
  if (n == 0) {
    *msg = (char *) Malloc(MallocSize_shortMessage);
    sprintf(*msg, "[1] digits expected in fft_parse()");
    return -1;
  }
 
  /* convert to (int) x */
  if (n > (int) strlen("1073741824")) { /* 2^30 */
    *x = -1;
  } else {
    *x = 0;
    for (i = 0; i < n; ++i)
      *x = 10*(*x) + ((int)(in[i]) - (int)'0');
    if (*x > 1073741824)
      *x = -1;
  }

  /* check size */
  if (!( (1 <= *x) && (*x <= 1073741824) )) {
    *msg = (char *) Malloc(MallocSize_shortMessage);
    sprintf(*msg, "[1] integer too large in fft_parse()");
    return -1;
  }
  return n;
}

static int parse_fft(char **msg, char *in, fft_t **F) {
  fft_t *Fq[fft_max_factors];   /* fft-object arguments */
  int    nq, q[fft_max_factors];   /* integer arguments */
  int    n,                /* nr. of chars read from in */
         N,                             /* length of *F */
         a;                          /* counter for q[] */
  char  *functor;              /* space for the functor */ 

#define isFunctor(f) \
  ((n == (int)strlen(f)) && (strncmp(in, f, n) == 0)) 

#define read_int(x) \
  { n += parse_int(msg, in+n, &x); \
    if (*msg != NULL) return -1; }

#define read_fft(F) \
  { n += parse_fft(msg, in+n, &(F)); \
    if (*msg != NULL) return -1; }

#define require(c) \
  { if (in[n] == (c)) { ++n; } else { \
    *msg = (char *) Malloc(MallocSize_shortMessage); \
    sprintf(*msg, "[1] '%c' expected in fft_parse()", c); return -1; } }

  /* read the functor up to '[' */
  n = 0;
  while ((in[n] != (char)0) && (in[n] != '['))
    ++n;

  /* read the expression */
  if (isFunctor("null")) {
    require('[');
    read_int(N);
    require(']');

    *msg = fft_applicable(fft_null, N, 0, NULL);
    if (*msg != NULL) return -1;

    *F = fft_new_null(N);
    return n;
  } else

  if (isFunctor("direct")) {
    require('[');
    read_int(N);
    require(']');

    *msg = fft_applicable(fft_direct, N, 0, NULL);
    if (*msg != NULL) return -1;

    *F = fft_new_direct(N);
    return n;
  } else

  if (isFunctor("small")) {
    require('[');
    read_int(N);
    require(']');

    *msg = fft_applicable(fft_small, N, 0, NULL);
    if (*msg != NULL) return -1;

    *F = fft_new_small(N);
    return n;
  } else

  if (isFunctor("rader")) {
    require('[');
    read_fft(Fq[0]);
    require(']');

    *msg = fft_applicable(fft_rader, (Fq[0]->N)+1, 0, NULL);
    if (*msg != NULL) return -1;

    *F = fft_new_rader(Fq[0]);
    return n;
  } else

  if (isFunctor("goodThomas")) {
    require('[');
    nq = 0;
    read_fft(Fq[0]); 
    ++nq;
    while (in[n] == ',') {
      if (nq == fft_max_factors) {
        *msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(*msg, 
          "[1] too many arguments of goodThomas[..] in fft_parse()"
        );
        while (nq > 0) {
          --nq;
          fft_delete(Fq[nq]);
        }
        return -1;
      }
      require(',');
      read_fft(Fq[nq]);
      ++nq;
    }
    require(']');

    N = 1;
    for (a = 0; a < nq; ++a) {
      q[a] = Fq[a]->N;
      N    = N * q[a];
    }
    *msg = fft_applicable(fft_goodThomas, N, nq, q);
    if (*msg != NULL) return -1;

    *F = fft_new_goodThomas(nq, Fq);
    return n;
  } else

  if (isFunctor("cooleyTukey")) {
    require('[');
    read_fft(Fq[0]); 
    require(',');
    read_fft(Fq[1]); 
    require(']');

    q[0] = Fq[0]->N;
    q[1] = Fq[1]->N;
    N    = q[0]*q[1];
    *msg = fft_applicable(fft_cooleyTukey, N, 2, q);
    if (*msg != NULL) return -1;

    *F = fft_new_cooleyTukey(Fq[0], Fq[1]);
    return n;
  } else

  if (isFunctor("qPower")) {
    require('[');
    read_fft(Fq[0]); 
    require(',');
    read_int(q[1]); 
    require(']');

    N = 1;
    for (a = 0; a < q[1]; ++a) {
      N = N * Fq[0]->N;
    }
    q[0] = Fq[0]->N;
    *msg = fft_applicable(fft_qPower, N, 2, q);
    if (*msg != NULL) return -1;

    *F = fft_new_qPower(Fq[0], q[1]);
    return n;
  } else

  if (isFunctor("bluestein")) {
    require('[');
    read_int(N); 
    require(',');
    read_fft(Fq[0]); 
    require(']');

    q[0] = Fq[0]->N;
    *msg = fft_applicable(fft_bluestein, N, 1, q);
    if (*msg != NULL) return -1;

    *F = fft_new_bluestein(N, Fq[0]);
    return n;
  } else

  if (isFunctor("simple")) {
    require('[');
    read_int(N);
    require(']');

    *msg = fft_applicable(fft_null, N, 0, NULL);
    if (*msg != NULL) return -1;

    *F = fft_new_simple(N);
    return n;
  } else

  if (isFunctor("prettyGood")) {
    require('[');
    read_int(N);
    require(']');

    *msg = fft_applicable(fft_null, N, 0, NULL);
    if (*msg != NULL) return -1;

    *F = fft_new_prettyGood(N);
    return n;
  } else

  /* unrecognized functor */
  functor = (char *) Malloc((n+1) * sizeof(char));
  strncpy(functor, in, n);
  functor[n] = (char)0;
  *msg = (char *) Malloc(MallocSize_shortMessage);
  sprintf(*msg, "[1] unrecognized functor \"%s\" in fft_parse()", functor);
  return -1;

#undef isFunctor
#undef read_int
#undef read_fft
#undef require
}

extern fft_t *fft_parse(char **msg, char *in) {
  fft_t *F;  /* the resulting structure */
  char  *in1; /* a processed copy of in */
  int    i;            /* index into in */

  /* strip comments and whitespace from in */
  in1 = (char *) Malloc( strlen(in) * sizeof(char) + 1 );
  i   = 0;
  while (*in != (char)0) {
    switch (*in) {
      case ' ':
      case '\t':
      case '\n':
        ++in;
        break;

      case '(':
        if (*(in+1) != '*') {
          in1[i++] = *(in++);
        } else {

          /* skip comment */
          do {
            ++in;
          } while (
            (*in != (char)0) && 
            ((*in != '*') || (*(in+1) != ')'))
          );
          if (*in == '*')
            in += 2;
        }
        break;

      default:
        in1[i++] = *(in++);
    }
  }
  in1[i] = (char)0;

  /* recursive descent parser */
  *msg  = NULL;
  F     = NULL;
  parse_fft(msg, in1, &F);
  Free(in1);
  if (*msg != NULL)
    return NULL;
  return F;
}

/* Test Applicability of a Specific FFT-Method
   ===========================================
*/

static char *check_baseField(char *method, int M, int N) {
  char  *msg;
  Scalar(dummy);

  if (!( (1 <= N) && (N <= fft_max_length) )) {
    msg = (char *) Malloc(MallocSize_shortMessage);
    sprintf(msg, 
      "[3] length N = %d must be in [1..fft_max_length] for fft_%s()", 
                      N,                                        method
    );
    return msg;
  }
  if (! fft_rational(dummy, 1, M)) {
    msg = (char *) Malloc(MallocSize_shortMessage);
    sprintf(msg, 
      "[2] base field must contain 1/%d for fft_%s()", 
                                     M,         method
    );
    return msg;
  }
  if (! fft_rootOfUnity(dummy, N, 1)) {
    msg = (char *) Malloc(MallocSize_shortMessage);
    sprintf(msg, 
      "[2] base field must contain %d-th roots of unity for fft_%s()", 
                                   N,                           method
    );
    return msg;
  }
  return NULL;
}

char *fft_applicable(int method, int N, int nq, int q[]) {
  char *msg;
  int   n, a, b;

  switch (method) {
    case fft_null:
      if (nq != 0) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_null, ..)", 
                                          nq
        );
        return msg;
      }
      return check_baseField("null", N, N);

    case fft_direct:
      if (nq != 0) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_direct, ..)", 
                                          nq
        );
        return msg;
      }
      return check_baseField("direct", N, N);

    case fft_small:
      if (nq != 0) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_small, ..)", 
                                          nq
        );
        return msg;
      }
      if (!( (1 <= N) && (N <= fft_max_small) )) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg,
          "[3] length N = %d must be in [1..fft_max_small] for fft_small()",
                          N
        );
        return msg;
      }
      msg = check_baseField("small", N, N);
      if (msg != NULL)
        return msg;
      switch (N) {
        case 1: break;
        case 2: break;
        case 3: return check_baseField("small3", 2, 2);
        case 4: break;
        case 5: return check_baseField("small5", 4, 2);
        case 6: break;
        case 7: return check_baseField("small5", 6, 2);
        case 8: break;
        case 9: return check_baseField("small5", 2, 2);
        default:
          Error("unrecognized N in fft_applicable(fft_small, ..)");
      }
      break;

    case fft_rader:
      if (nq != 0) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_rader, ..)", 
                                          nq
        );
        return msg;
      }
      msg = check_baseField("rader", N, N);
      if (msg != NULL)
        return msg;
      if (! primeQ(N)) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[4] length N = %d must be prime for fft_rader()",
                          N
        );
        return msg;
      }
      return check_baseField("rader", N-1, N-1);

    case fft_goodThomas:
      if (!( (2 <= nq) && (nq <= fft_max_factors) )) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_goodThomas, ..)", 
                                          nq
        );
        return msg;
      }
      n = 1;
      for (a = 0; a < nq; ++a) {
        if (!( (2 <= q[a]) && (q[a] <= fft_max_length) )) {
          msg = (char *) Malloc(MallocSize_shortMessage);
          sprintf(msg, 
            "[3] length q[%d] = %d must be in [1..fft_max_length]"
            " for fft_goodThomas()", 
                          a,    q[a]
          );
          return msg;
        }
        n = n * q[a];
        if (!( (2 <= n) && (n <= fft_max_length) )) {
          msg = (char *) Malloc(MallocSize_shortMessage);
          sprintf(msg, 
            "[3] length must be in [1..fft_max_length]"
            " for fft_goodThomas()"
          );
          return msg;
        }
      }
      if (N != n) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] length N = %d must be q[0]* .. *q[nq-1] for fft_goodThomas()",
                          N
        );
        return msg;        
      }
      for (a = 0; a < nq-1; ++a)
        for (b = a+1; b < nq; ++b)
          if (gcd(q[a], q[b]) != 1) {
            msg = (char *) Malloc(MallocSize_shortMessage);
            sprintf(msg, 
              "[5] q[%d] = %d and q[%d] = %d must be coprime"
              " for fft_goodThomas()",
                     a,    q[a],    b,    q[b]
            );
            return msg;
          }
      return check_baseField("goodThomas", N, N);

    case fft_cooleyTukey:
      if (nq != 2) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_cooleyTukey, ..)", 
                                          nq
        );
        return msg;
      }
      if (!( 
        (2 <= q[0]) && (q[0] <= fft_max_length) &&
        (2 <= q[1]) && (q[1] <= fft_max_length)
      )) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[3] q[0] = %d and q[1] = %d must be in [1..fft_max_length] "
          "for fft_cooleyTukey()", 
                      q[0],         q[1]
        );
        return msg;
      }
      if (N != q[0]*q[1]) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] length N = %d must be q[0]*q[1] for fft_cooleyTukey()",
                          N
        );
        return msg;        
      }
      return check_baseField("cooleyTukey", N, N);

    case fft_qPower:
      if (nq != 2) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_qPower, ..)", 
                                          nq
        );
        return msg;
      }
      if (!( (2 <= q[0]) && (q[0] <= fft_max_length) )) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[3] radix q[0] = %d must be in [1..fft_max_length] "
          "for fft_qPower()",
                            q[0] 
        );
        return msg;
      }
      if (!( (2 <= q[1]) && (q[1] <= 32) )) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[3] exponent q[1] = %d must be in [1..Log(2, fft_max_length)] "
          "for fft_qPower()",
                               q[1] 
        );
        return msg;
      }
      n = 1;
      for (a = 0; a < q[1]; ++a) {
        n = n * q[0];
        if (!( (2 <= n) && (n <= fft_max_length) )) {
          msg = (char *) Malloc(MallocSize_shortMessage);
          sprintf(msg, 
            "[3] length must be in [1..fft_max_length] for fft_qPower()"
          );
          return msg;
        }
      }
      if (N != n) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] length N = %d must be q[0]^q[1] for fft_qPower()",
                          N
        );
        return msg;        
      }
      return check_baseField("qPower", N, N);

    case fft_bluestein:
      if (nq != 1) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[7] wrong number of arguments (%d) "
          "for fft_applicable(fft_bluestein, ..)", 
                                          nq
        );
        return msg;
      }
      msg = check_baseField("bluestein", 2*N, 2*N);
      if (msg != NULL)
        return msg;
      if (!( (2*N <= q[0]) && (q[0] <= fft_max_length) )) {
        msg = (char *) Malloc(MallocSize_shortMessage);
        sprintf(msg, 
          "[6] embedding q[0] = %d must be >= 2*N = %d for fft_bluestein()",
                                q[0],               2*N
        );
        return msg;
      }
      return check_baseField("bluestein", q[0], q[0]);

    default:
      msg = (char *) Malloc(MallocSize_shortMessage);
      sprintf(msg, "[8] unrecognized method = %d in fft_applicable()",
                                              method
      );
      return msg;
  }
  return NULL;
}
