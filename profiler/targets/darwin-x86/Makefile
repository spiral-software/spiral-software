##
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details
##
# do not consider the standard implicit suffixes
.SUFFIXES: 

# these are defines that can be passed in

CC              ?= gcc
CFLAGS          ?= -O2
LDFLAGS         ?= -lm
OUTDIR          ?= .
GAP             ?= $(OUTDIR)/testcode.c
ADDSRC          ?=
STUB            ?= $(OUTDIR)/testcode.h
TIMER           ?= ../common/rdtsc_time.c
TIMER_OPTS      ?= -n 100
MATRIX          ?= ../common/matrix.c
VECTOR          ?= ../common/cvector.c

COUT   ?= $(OUTDIR)/cout.txt
CERR   ?= $(OUTDIR)/cerr.txt

# setup some defaults.
export RM      = rm -f
export AR      = ar
export RANLIB  = ranlib

export CC

override CFLAGS += -w -I. -I.. -I../common -I../conf -I../../.. -include $(STUB) -DHAVE_UNISTD_H -DINSIDE_SPIRAL

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
all: clean buildTimer testTimer

.PHONY: build
build: buildTimer

.PHONY: test
test: testTimer

.PHONY: matrix
matrix: clean buildMatrix testMatrix

.PHONY: vector
vector: clean buildVector testVector


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
	$(MAKE) -C ../conf -f Makefile.linux CONFDIR=$(CURDIR)

##########
# rules for .c -> .o
###########

define C_TO_OBJ
$(OUTDIR)/$(notdir $(1:.c=.o)): $(1)
	$(CC) $(CFLAGS) -c -o $$@ $$<  >$(COUT) 2>$(CERR)
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

ifeq (,$(findstring $(CC),"icc gcc"))
%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

-include $(TIMER:.c=.d)
-include $(MATRIX:.c=.d)
-include $(VECTOR:.c=.d)
-include $(SRC:.c=.d)
endif

# hardcoded dependency information. this is required such that the timer is rebuilt when the stub changes.

$(OUTDIR)/$(notdir $(TIMER:.c=.o)): $(STUB)
$(OUTDIR)/$(notdir $(MATRIX:.c=.o)): $(STUB)
$(OUTDIR)/$(notdir $(VECTOR:.c=.o)): $(STUB)

# cleanup
###########

TIMEREXECLEAN = $(TIMEREXE)
MATRIXEXECLEAN = $(MATRIXEXE)
VECTOREXECLEAN = $(VECTOREXE)

.PHONY: clean
clean: 
	$(MAKE) -C ../conf -f Makefile.linux CONFDIR=$(CURDIR) clean
	-$(RM) $(SRC:.c=.d) $(TIMER:.c=.d) $(OBJ) $(TIMEROBJ) $(MATRIXOBJ) $(VECTOROBJ) $(TIMEREXECLEAN) $(MATRIXEXECLEAN) $(VECTOREXECLEAN)

