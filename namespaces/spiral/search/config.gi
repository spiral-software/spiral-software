
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


# Configuration of system dependent parameters
# ============================================
# and global variables
# ====================
# MP, BS, from 01/25/01, GAPv3.4.4


# Search Options and Global Variables
# -----------------------------------

# time limit
SEARCH_TIME_LIMIT := false;

# should search algorithms search over local unrolling?
# and when they do so, what is the smallest and largest sizes to try
SEARCH_LOCAL_UNROLLING := false;
SEARCH_LOCAL_UNROLLING_MIN := 4;
SEARCH_LOCAL_UNROLLING_MAX := 64;

# should search algorithms search over global unrolling?
# and when they do so, what is the smallest and largest sizes to try
SEARCH_GLOBAL_UNROLLING := false;
SEARCH_GLOBAL_UNROLLING_MIN := 8;
SEARCH_GLOBAL_UNROLLING_MAX := 64;

# should the search algorithms save their results to the BestFoundTable?
SEARCH_BEST_FOUND := "save";

# how many best formulas should DP keep?
DP_N_BEST := 1;
# should we try to minimize or maximize?
DP_OPTIMIZE := "minimize";
# how verbose should DP be?
DP_VERBOSITY := 3;
# the default veclen for DPVec
DPVEC_VLEN := 4;
# default value for save memory
DP_SAVEMEMORY := true;


# Exhaustive Search options:
EXHAUSTIVE_SEARCH_VERBOSITY := 2;

# Timed Search options:
TIMED_SEARCH_TIME_LIMIT := 30;
TIMED_SEARCH_SEARCHES := [
   "DP", rec(), false,
   "DP", rec(localUnrolling:=true,nBest:=4), false,
		         ];
TIMED_SEARCH_VERBOSITY := 3;

# Implement options:
IMPLEMENT_SEARCH := "BestOrTimedSearch";
IMPLEMENT_SEARCH_OPTS := rec();
IMPLEMENT_SPLOPTS := rec();
IMPLEMENT_VERBOSITY := 3;


# BestFoundDifferFields is a list of SPLOptions fields for which the
# BestFoundTable should maintain seperate best found implementations
BestFoundDifferFields := [ "dataType", "language" ];



# Defaults derived from parameters above
# --------------------------------------

# global search options record
SEARCH_DEFAULTS :=
  rec(
    timeLimit := SEARCH_TIME_LIMIT,
    localUnrolling := SEARCH_LOCAL_UNROLLING,
    localUnrollingMin := SEARCH_LOCAL_UNROLLING_MIN,
    localUnrollingMax := SEARCH_LOCAL_UNROLLING_MAX,
    globalUnrolling := SEARCH_GLOBAL_UNROLLING,
    globalUnrollingMin := SEARCH_GLOBAL_UNROLLING_MIN,
    globalUnrollingMax := SEARCH_GLOBAL_UNROLLING_MAX,
    bestFound := SEARCH_BEST_FOUND,
    timeBaseCases := true
  );

# global DP options record
DP_DEFAULTS :=
   rec(
      nBest := DP_N_BEST,
      optimize := DP_OPTIMIZE,
      verbosity := DP_VERBOSITY,
	  DPVec := false,
	  DPVecVlen := DPVEC_VLEN,
      # setting this to true is dangerous, it will do var.flush() on every measurement
      saveMemory := false,
   );


# global Exhaustive Search options record
EXHAUSTIVE_SEARCH_DEFAULTS :=
   rec(
      verbosity := EXHAUSTIVE_SEARCH_VERBOSITY
   );

# global Exhaustive Search options record
TIMED_SEARCH_DEFAULTS :=
   rec(
      timeLimit := TIMED_SEARCH_TIME_LIMIT,
      searches  := TIMED_SEARCH_SEARCHES,
      verbosity := TIMED_SEARCH_VERBOSITY
   );

# global Implement options record
IMPLEMENT_DEFAULTS :=
   rec(
      search     := IMPLEMENT_SEARCH,
      searchOpts := IMPLEMENT_SEARCH_OPTS,
      SPLOpts    := IMPLEMENT_SPLOPTS,
      verbosity  := IMPLEMENT_VERBOSITY
   );


# Important global variables
# --------------------------

# BestFoundTable
# - contains the best found implementations
# - set in best.g


# Global auxiliary variables
# --------------------------

savedHashTable := "Problems Reading HashTable";  # (used in hash.g)
savedBestFoundTable := "Problems Reading BestFoundTable";  # (used in best.g)

