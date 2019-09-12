##
##  Copyright (c) 2018-2019, Carnegie Mellon University
##  See LICENSE for details
##
# do not consider the standard implicit suffixes
.SUFFIXES: 

# these are defines that can be passed in

CC              ?= cl.exe
OUTDIR 		?= .
GAP    		?= $(OUTDIR)/testcode.c
STUB   		?= $(OUTDIR)/testcode.h
TIMER  		?= ../common/rdtsc_time.c
MATRIX 		?= ../common/matrix.c
COUT   		?= $(OUTDIR)/cout.txt
CERR   		?= $(OUTDIR)/cerr.txt
LLOG		?= $(OUTDIR)/llog.txt

# setup some defaults.
export SHELL   = cmd.exe
export RM      = del /f /q
export LINK    = link

export CC

override CFLAGS  = /I.. /I. /I..\common /I..\conf /FI$(subst /,\,$(STUB)) /DWIN64 /nologo /Dinline=__inline /O2 /Oi /openmp
override LFLAGS  = /nologo /subsystem:console /incremental:no /PDB:$(subst /,\,$(OUTDIR))\vc70.pdb

# this controls the output directory for the conf stuff.
export CONFDIR = ../win-x64-vcc

TIMEREXE=$(OUTDIR)/$(notdir $(subst .c,.exe,$(TIMER)))
MATRIXEXE=$(OUTDIR)/$(notdir $(subst .c,.exe,$(MATRIX)))

# some defines for source and object files.
######

# object files are always placed into the local directory.
OBJ=$(addprefix $(OUTDIR)/, $(notdir $(GAP:.c=.obj)))
TIMEROBJ=$(addprefix $(OUTDIR)/, $(notdir $(TIMER:.c=.obj)))
MATRIXOBJ=$(addprefix $(OUTDIR)/, $(notdir $(MATRIX:.c=.obj)))

ALLSRC = $(TIMER) $(MATRIX) $(GAP)

# the make commands: all, build, test
.PHONY: all
all: build test 

.PHONY: build
build: delTimerBinFiles link.obj $(TIMEREXE) 

.PHONY: test
test: build
	@-$(TIMEREXE)

.PHONY: matrix
matrix: cleanMatrix buildMatrix testMatrix

.PHONY: buildMatrix
buildMatrix: delMatrixBinFiles link.obj $(MATRIXEXE)

.PHONY: testMatrix
testMatrix: buildMatrix
	@-$(MATRIXEXE)

.PHONY:	cleanMatrix
	-$(RM) $(subst /,\,$(MATRIXOBJ)) *.obj $(subst /,\,$(MATRIXEXE)) 2> nul

# the link command, mashes all objects together.
$(TIMEREXE): $(TIMEROBJ) $(OBJ) libconf.lib
	$(LINK) $(LFLAGS) /out:$(subst /,\,$@) $(subst /,\,$^)

$(MATRIXEXE): $(MATRIXOBJ) $(OBJ) libconf.lib
	$(LINK) $(LFLAGS) /out:$(subst /,\,$@) $(subst /,\,$^)

# make sure that timed code always up-to-date
delTimerBinFiles:
	-$(RM) $(OBJ) $(TIMEROBJ) 2> nul
	-$(RM) $(TIMEREXE) 2> nul

delMatrixBinFiles:
	-$(RM) $(OBJ) $(TIMEROBJ) 2> nul
	-$(RM) $(MATRIXEXE) 2> nul

libconf.lib:
	$(MAKE) -C ..\conf -f Makefile.win CONFDIR=$(CURDIR)

# the link.c file is empty but crucial. It is required because of 
# peculiarities in the WinAPI CreateProcess() call inside of make.exe.

# namely, there is a discrepancy between the 1st and 2nd param passed
# to CreateProcess, as the 1st contains the path-expanded version of 'link'
# and the 2nd has 'link' with all of its command line options. The 
# linker assumes the 2nd param consists only of command line arguments, 
# and interprets 'link' as a missing 'link.obj'. Yes, this sucks.
link.obj: link.c
	$(CC) /c /nologo /Fo$@ $< >$(LLOG)

##########
# rules for .c -> .o
###########

# this implicit rule does not work since the src/obj are not in the same dir 
# in some cases.
# %.obj: %.c
#	$(CC) $(CFLAGS) /c /Fo$@ $<

# instead we use the power of gnu make to autogenerate the .c/.o relationship
# for each file. refer to http://www.gnu.org/software/make/manual/make.pdf

# we pipe to null because visual c likes to spit out the name of the file
# it is compiling, and I'm not sure how to make it stop

define C_TO_OBJ
$(OUTDIR)/$(notdir $(1:.c=.obj)): $(1)
	$(CC) $(CFLAGS) /c /Fo$$(subst /,\,$$@) $$(subst /,\,$$<) >$(COUT) 2>$(CERR)
endef
$(foreach src,$(ALLSRC),$(eval $(call C_TO_OBJ, $(src))))

# hardcoded dependency information. this is required such that the timer is rebuilt when the stub changes.
$(OUTDIR)/$(notdir $(TIMER:.c=.obj)): $(STUB)


# cleanup 
#########

.PHONY: clean
clean: delTimerBinFiles delMatrixBinFiles
	$(MAKE) -C ..\conf -f Makefile.win CONFDIR=$(CURDIR) clean
