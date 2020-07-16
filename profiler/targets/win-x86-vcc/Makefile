##
##  Copyright (c) 2018-2020, Carnegie Mellon University
##  See LICENSE for details
##
# do not consider the standard implicit suffixes
.SUFFIXES: 

# these are defines that can be passed in

RM      = del /f /q
CC      = cl
MAKE	= nmake
LINK	= link

##  We expect OUTDIR to be passed as a command line argument

GAP    		= $(OUTDIR)\testcode.c
STUB   		= $(OUTDIR)\testcode.h
TIMERSRC  	= rdtsc_time.c
MATRIXSRC	= matrix.c
VECTORSRC	= cvector.c
COMMONDIR	= ..\..\targets\common
TIMER		= $(COMMONDIR)\$(TIMERSRC)
MATRIX		= $(COMMONDIR)\$(MATRIXSRC)
VECTOR		= $(COMMONDIR)\$(VECTORSRC)
COUT   		= $(OUTDIR)\cout.txt
CERR   		= $(OUTDIR)\cerr.txt
LLOG		= $(OUTDIR)\llog.txt

CFLAGS  = /I..\..\targets /I..\..\targets\common /I..\..\targets\conf /FI$(STUB) /DWIN64 /nologo /Dinline=__inline /O2 /Oi /openmp
LFLAGS  = /nologo /subsystem:console /incremental:no /PDB:$(OUTDIR)\vc70.pdb

# this controls the output directory for the conf stuff.
CONFDIR 	= ..\..\targets\win-x86-vcc

TIMEREXE=$(OUTDIR)\$(TIMERSRC:.c=.exe)
MATRIXEXE=$(OUTDIR)\$(MATRIXSRC:.c=.exe)
VECTOREXE=$(OUTDIR)\$(VECTORSRC:.c=.exe)

# object files are always placed into the local directory.
OBJ=$(GAP:.c=.obj)
TIMEROBJ=$(TIMEREXE:.exe=.obj)
MATRIXOBJ=$(MATRIXEXE:.exe=.obj)
VECTOROBJ=$(VECTOREXE:.exe=.obj)

ALLSRC = $(TIMER) $(MATRIX) $(VECTOR) $(GAP)

# the make commands: all, build, test
.PHONY: all
all: build test 

.PHONY: build
build: delTimerBinFiles $(TIMEREXE)

.PHONY: test
test: build
	@-$(TIMEREXE)

.PHONY: matrix
matrix: delMatrixBinFiles buildMatrix testMatrix

.PHONY: buildMatrix
buildMatrix: delMatrixBinFiles $(MATRIXEXE)

.PHONY: testMatrix
testMatrix: buildMatrix
	@-$(MATRIXEXE)

.PHONY: vector
vector: delVectorBinFiles buildVector testVector

.PHONY: buildVector
buildVector: delVectorBinFiles $(VECTOREXE)

.PHONY: testVector
testVector: buildVector
	@-$(VECTOREXE)

# the link command, mashes all objects together.
$(TIMEREXE): $(TIMEROBJ) $(OBJ) libconf.lib
	$(LINK) $(LFLAGS) /out:$@ $(TIMEROBJ) $(OBJ) $(CONFDIR)\libconf.lib

$(MATRIXEXE): $(MATRIXOBJ) $(OBJ) libconf.lib
	$(LINK) $(LFLAGS) /out:$@ $(MATRIXOBJ) $(OBJ) $(CONFDIR)\libconf.lib

$(VECTOREXE): $(VECTOROBJ) $(OBJ) libconf.lib
	$(LINK) $(LFLAGS) /out:$@ $(VECTOROBJ) $(OBJ) $(CONFDIR)\libconf.lib

# make sure that timed code always up-to-date
delTimerBinFiles:
	-$(RM) $(OBJ) $(TIMEROBJ) 2> nul
	-$(RM) $(TIMEREXE) 2> nul

delMatrixBinFiles:
	-$(RM) $(OBJ) $(MATRIXOBJ) 2> nul
	-$(RM) $(MATRIXEXE) 2> nul

delVectorBinFiles:
	-$(RM) $(OBJ) $(VECTOROBJ) 2> nul
	-$(RM) $(VECTOREXE) 2> nul

libconf.lib:
	$(MAKE) /C /S /f ..\..\targets\conf\Makefile.win CONFDIR=$(CONFDIR) all

##########
# rules for .c -> .o
###########

# Implicit rules for src/obj in nmake don't work as src and object are not in
# same directories.  NOTE: We can't simply build in the source directory as
# the source remains common for multiple targets.  Thus we have separate
# rules for each object (nmake doesn't support anything other than literal
# substitution and doesn't provide looping or rule definitions).

# we pipe to null because visual c likes to spit out the name of the file
# it is compiling, and I'm not sure how to make it stop

$(OBJ): $(GAP) $(STUB)
	$(CC) $(CFLAGS) /c /Fo$@ $(GAP) >$(COUT) 2>$(CERR)

$(TIMEROBJ): $(TIMER)
	$(CC) $(CFLAGS) /c /Fo$@ $(TIMER) >$(COUT) 2>$(CERR)

$(MATRIXOBJ):$(MATRIX)
	$(CC) $(CFLAGS) /c /Fo$@ $(MATRIX) >$(COUT) 2>$(CERR)

$(VECTOROBJ):$(VECTOR)
	$(CC) $(CFLAGS) /c /Fo$@ $(VECTOR) >$(COUT) 2>$(CERR)


# hardcoded dependency information. this is required such that the timer is rebuilt when the stub changes.
$(TIMEREXE:.exe=.obj): $(STUB)


# cleanup 
#########

.PHONY: clean
clean: delTimerBinFiles delMatrixBinFiles delVectorBinFiles
	$(MAKE) /C /S /f ..\..\targets\conf\Makefile.win CONFDIR=$(CONFDIR) clean
