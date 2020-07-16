##
##  Copyright (c) 2018-2020, Carnegie Mellon University
##  See LICENSE for details
##
# do not consider the standard implicit suffixes
.SUFFIXES: 

# these are defines that can be passed in

CC              ?= clang
CFLAGS          ?= -O2
LDFLAGS         ?= 
OUTDIR          ?= .
GAP             ?= $(OUTDIR)/testcode.c
ADDSRC          ?=
STUB            ?= $(OUTDIR)/testcode.h
TIMER           ?= ../common/rdtsc_time.c
TIMER_OPTS      ?= -n 100
MATRIX          ?= ../common/matrix.c
VECTOR          ?= ../common/cvector.c

# setup some defaults.
export RM      = del /F /Q
export AR      = llvm-ar
export RANLIB  = llvm-ranlib

export CC

override CFLAGS += -fopenmp -I.. -I. -I../common -I../conf -I../../.. -include $(STUB) -DWIN32 -DWIN64

# this is our executable.
TIMEREXE = $(OUTDIR)/$(notdir $(subst .c,,$(TIMER)))
MATRIXEXE = $(OUTDIR)/$(notdir $(subst .c,,$(MATRIX)))
VECTOREXE = $(OUTDIR)/$(notdir $(subst .c,,$(VECTOR)))

# some defines for source and object files.
######

SRC:=$(GAP) $(ADDSRC)

# objects placed in our output dir.
OBJ  =$(addprefix $(OUTDIR)/, $(notdir $(SRC:.c=.o)))
TIMEROBJ  =$(addprefix $(OUTDIR)/, $(notdir $(TIMER:.c=.o)))

# supported make commands: all, build, test
.PHONY: all
all: buildTimer testTimer

.PHONY: build
build: buildTimer

.PHONY: test
test: testTimer

.PHONY: matrix
matrix: buildMatrix testMatrix

.PHONY: vector
vector: buildVector testVector


# internals of this Makefile
.PHONY: buildTimer
buildTimer: $(TIMEREXE) 

.PHONY: buildMatrix
buildMatrix: $(MATRIXEXE)

.PHONY: buildVector
buildVector: $(VECTOREXE)

.PHONY: testTimer
testTimer: buildTimer
	@-$(TIMEREXE) $(TIMER_OPTS)

.PHONY: testMatrix
testMatrix: buildMatrix
	@-$(MATRIXEXE)

.PHONY: testVector
testVector: buildVector
	@-$(VECTOREXE)

# the link command, mashes all objects together.
$(TIMEREXE): $(TIMER) $(OBJ) libconf.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(MATRIXEXE): $(MATRIX) $(OBJ) libconf.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(VECTOREXE): $(VECTOR) $(OBJ) libconf.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# makes the libconf library in the ../conf dir.
# this really isn't PHONY but PHONY forces the execution of the make
# command to confirm that the libconf.a library is actually up to date.
.PHONY: libconf.a
libconf.a:
	$(MAKE) -C ../conf -f Makefile.llvm CONFDIR=$(CURDIR)

##########
# rules for .c -> .o
###########

define C_TO_OBJ
$(OUTDIR)/$(notdir $(1:.c=.o)): $(1)
	$(CC) $(CFLAGS) -c -o $$@ $$< 1>&2
endef

#${OBJ}: ${SRC}
#	$(CC) $(CFLAGS) -c -o $@ $< 1>&2

$(foreach src,$(SRC),$(eval $(call C_TO_OBJ, $(src))))
$(foreach src,$(TIMER),$(eval $(call C_TO_OBJ, $(src))))
$(foreach src,$(MATRIX),$(eval $(call C_TO_OBJ, $(src))))
$(foreach src,$(VECTOR),$(eval $(call C_TO_OBJ, $(src))))

# implicit rules that build .h file dependecy information using the gcc -MM feature of gcc. the .d files are then
# included in the makefile.
##########


# hardcoded dependency information. this is required such that the timer is rebuilt when the stub changes.

$(OUTDIR)/$(notdir $(TIMER:.c=.o)): $(STUB)

# cleanup
###########


TIMEREXECLEAN = $(TIMEREXE)
MATRIXEXECLEAN = $(MATRIXEXE)
VECTOREXECLEAN = $(VECTOREXE)


.PHONY: clean
clean: 
	$(MAKE) -C ../conf -f Makefile.llvm CONFDIR=$(CURDIR) clean
	-$(RM) $(subst /,\,$(SRC:.c=.d)) $(subst /,\,$(TIMER:.c=.d))  \
	$(subst /,\,$(OBJ))) $(subst /,\,$(TIMEROBJ))  \
	$(subst /,\,$(TIMEREXECLEAN)) $(subst /,\,$(MATRIXEXECLEAN)) $(subst /,\,$(VECTOREXECLEAN)) 2> null
