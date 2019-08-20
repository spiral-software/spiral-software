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
TIMER           ?= timer.c
TIMER_OPTS      ?= -n 100

# setup some defaults.
export RM      = rm -f
export AR      = ar
export RANLIB  = ranlib

export CC

override CFLAGS += -I. -I../common -I../conf -I../../.. -include $(STUB) -DHAVE_UNISTD_H

SYSTEM = $(shell uname -s)

# this is our executable.
TIMEREXE = $(OUTDIR)/$(notdir $(subst .c,,$(TIMER)))

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


# internals of this Makefile
.PHONY: buildTimer
buildTimer: $(TIMEREXE) 

.PHONY: testTimer
testTimer: buildTimer
	@$(TIMEREXE) $(TIMER_OPTS)

# the link command, mashes all objects together.
$(TIMEREXE): $(TIMER) $(OBJ) libconf.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


# makes the libconf library in the ../conf dir.
# this really isn't PHONY but PHONY forces the execution of the make
# command to confirm that the libconf.a library is actually up to date.
.PHONY: libconf.a
libconf.a:
	$(MAKE) -C ../conf -f Makefile.darwin CONFDIR=$(CURDIR)

##########
# rules for .c -> .o
###########

define C_TO_OBJ
$(OUTDIR)/$(notdir $(1:.c=.o)): $(1)
	$(CC) $(CFLAGS) -c -o $$@ $$< 1>&2
endef

$(foreach src,$(SRC),$(eval $(call C_TO_OBJ, $(src))))
$(foreach src,$(TIMER),$(eval $(call C_TO_OBJ, $(src))))

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
-include $(SRC:.c=.d)
endif

# hardcoded dependency information. this is required such that the timer is rebuilt when the stub changes.

$(OUTDIR)/$(notdir $(TIMER:.c=.o)): $(STUB)

# cleanup
###########

ifeq (Cygwin,$(SYSTEM))
TIMEREXECLEAN = $(addsuffix .exe,$(TIMEREXE))
else
TIMEREXECLEAN = $(TIMEREXE)
endif

.PHONY: clean
clean: 
	$(MAKE) -C ../conf -f Makefile.linux CONFDIR=$(CURDIR) clean
	-$(RM) $(SRC:.c=.d) $(TIMER:.c=.d) $(TIMEROBJ) $(TIMEREXECLEAN) 
