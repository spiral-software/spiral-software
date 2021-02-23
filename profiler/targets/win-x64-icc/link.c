/*
 *  Copyright (c) 2018-2021, Carnegie Mellon University
 *  See LICENSE for details
 */
 
/*
 *  The link.c file is empty but crucial. It is required because of
 *  peculiarities in the WinAPI CreateProcess() call inside of make.exe.
 *
 *  Namely, there is a discrepancy between the 1st and 2nd param passed to
 *  CreateProcess, as the 1st contains the path-expanded version of 'link' and
 *  the 2nd has 'link' with all of its command line options. The linker
 *  assumes the 2nd param consists only of command line arguments, and
 *  interprets 'link' as a missing 'link.obj'. Yes, this sucks.
 */
