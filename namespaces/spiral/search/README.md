Search Module
=============

The search module uses different
search algorithms to find fast implementations of transforms.

- ExhaustiveSearch() -- measure all possible ruletrees.
- DP()               -- performs dynamic programming.
- RandomSearch()     -- generates random ruletrees, searching for the fastest.
- HillClimb()        -- performs hill climbing search.
- STEER()            -- a stochastic evolutionary search algorithm (similar to a genetic algorithm).
- TimedSearch()      -- a meta-search algorithm that calls others with specific time limits.

General Search Options
----------------------
 
These search options are common to many of the algorithms:
```
  timeLimit := false | <minutes>,
  localUnrolling := true | false,
  localUnrollingMin := <positive int>,
  localUnrollingMax := <positive int>,
  globalUnrolling := true | false,
  globalUnrollingMin := <positive int>,
  globalUnrollingMax := <positive int>,
  bestFound := "save" | "none"
```

Time limit specifies the maximum number of minutes that a search 
algorithm should continue searching.  Setting this to false allows the
search algorithm to continue searching until it is finished.  Most, but
not all, search algorithms (notably not DP) can give back a reasonable
result after any amount of time.  Since GAP does not have an interrupt
mechanism, it is quite possible for a search algorithm to take
considerably more time than specified.  A check on time is only
conducted at certain points in the loops/recursion of the search
algorithms.

Local unrolling specifies to the SPL compiler exactly which portions of
the ruletree are to be unrolled.  If a specific node in the ruletree is
marked for unrolling, then the SPL compiler will unroll the code for
that node and its subtree.

Global unrolling specifies to the SPL compiler that ALL nodes of the
given size and smaller are to be unrolled.

Setting localUnrolling to false causes the search algorithms to only
search through ruletrees without any nodes marked for unrolling.
Setting localUnrolling to true causes the search algorithms to search
through the space of ruletrees with nodes marked for unrolling.
Specifically, all nodes of size localUnrollingMin and smaller are marked
for unrolling, while those of size greater than localUnrollingMin and less
than or equal to localUnrollingMax are considered both with and without
unrolling.  Turning on search over local unrolling turns off all global
unrolling specified to the SPL compiler.

Setting globalUnrolling to true causes the search algorithms to search
over different settings for the global unrolling.  Specifically they
search from globalUnrollingMin to globalUnrollingMax, inclusive, going
by factors of 2 (that is, they consider globalUnrollingMin,
globalUnrollingMin*2, globalUnrollingMin*4, etc ... until
globalUnrollingMin*2^k > globalUnrollingMax).

It is not allowable to search over both local and global unrolling.

Setting bestFound to "save" causes the search modules to try to save
their best found implementations for spls into the BestFoundTable.
If they did not find one faster than previously found, then nothing
happens.  Setting bestFound to "none" causes the search modules to do
nothing with the BestFoundTable.


ExhaustiveSearch
----------------
 
Usage: ```ExhaustiveSearch( <spl> [, <Exhaustive-Search-options-record>, <SPLOptionsRecord> ] )```

Note that if you wish to specify an Exhaustive-options-record or a
SPL-options-record, then you must specify both.  To leave one blank,
just pass "rec()" in its place.

Searches over all possible ruletrees and returns the one with the
fastest runtime.  Currently does NOT search over local or global unrolling
parameters, just different ruletrees.

Returns:  fastest ruletree

### Options
```
spiral> PrintSpecExhaustiveSearchOptionsRecord();             
rec(
  timeLimit := false | <minutes>,
  bestFound := "save" | "none",
  verbosity := <non-negative integer>
);
```
Run ```MergeExhaustiveSearchOptionsRecord( rec() );``` to determine
defaults.

### Verbosity Levels
```
   0 = Print nothing
   1 = Pretty print final best tree
   2 = Also print how many measurements to do
   3 = Print all trees as measured
```

### Examples
```
spiral> W := SPLNonTerminal("WHT", 3);;
spiral> ExhaustiveSearch(W);
#I no. trees: 1 3 3 3 3 3 
#I 3 measurements to do

Best Rule Tree:
WHT(3)     {RuleWHT_1}
 |--WHT(1)     {RuleWHT_0}
 `--WHT(2)     {RuleWHT_1}
     |--WHT(1)     {RuleWHT_0}
     `--WHT(1)     {RuleWHT_0}
 ! 215

RuleTree(
  RuleWHT_1,
  SPLNonTerminal( "WHT", 3 ), [
  RuleTree(RuleWHT_0, SPLNonTerminal( "WHT", 1 )),
  RuleTree(
    RuleWHT_1,
    SPLNonTerminal( "WHT", 2 ), [
    RuleTree(RuleWHT_0, SPLNonTerminal( "WHT", 1 )),
    RuleTree(RuleWHT_0, SPLNonTerminal( "WHT", 1 ))
  ] )
] )
spiral> ExhaustiveSearch( W, rec(verbosity:=1), rec(language:="c") );;
#I no. trees: 1 3 3 3 3 3 

Best Rule Tree:
WHT(3)     {RuleWHT_1}
 |--WHT(1)     {RuleWHT_0}
 `--WHT(2)     {RuleWHT_1}
     |--WHT(1)     {RuleWHT_0}
     `--WHT(1)     {RuleWHT_0}
 ! 216

spiral> 
```

DP (Dynamic Programming)
------------------------

Usage: ```DP( <spl> [, <DP-options-record>, <SPL-options-record> ] )```

Note that if you wish to specify a DP-options-record or a SPL-options-record,
then you must specify both.  To leave one blank, just pass "rec()" in its
place.

DP() runs dynamic programming on the given spl.  Specifically, dynamic
programming considers all applicable rules to the given transform, and
all possible sets of children generated by those rules.  For each child,
it looks up in its table the fastest implementation of that transform.
If no such entry exists yet in its table, then DP recursively calls
itself on the child.  Once DP has found the best implementation for the
child, it substitutes this ruletree in as a subtree of the root in place
of the child.  DP then times all such trees, determines the fastest one,
and enters that in its table.

Returns: List of records.  Each record contains a ruletree and the measured
time for that ruletree.  Fastest ruletree is first entry in the list.  The
list has <nBest> many entries, from the fastest to the nBest fastest formulas
found.

### Options
```
spiral> PrintSpecDPOptionsRecord();
rec(
  timeLimit := false | <minutes>,
  localUnrolling := true | false,
  localUnrollingMin := <positive int>,
  localUnrollingMax := <positive int>,
  globalUnrolling := true | false,
  globalUnrollingMin := <positive int>,
  globalUnrollingMax := <positive int>,
  bestFound := "save" | "none",
  nBest := <positive integer>,
  optimize := "minimize" | "maximize",
  hashTable := <hashTable>,
  verbosity := <non-negative integer>
);
```
Run 'MergeDPOptionsRecord( rec() );' to determine defaults.

An nBest of 1 is the "standard" DP.  Increasing nBest causes DP to not
only keep in its list the best implementation for each spl and size,
but the nBest such implementations.

By setting optimize to "maximize" it is possible to cause DP to maximize
the measured event instead of the usual minimization.

DP uses a hash table to store its list of best implementations.  It is
possible to pass a hash table to DP for it to use and so that you can
keep the resulting hash table when DP finishes.  Note that any entries
in the hashTable will be assumed to the best implementations by DP.
You can create a new hashTable for use with DP by calling HashTableDP().

### Verbosity Levels
```
   0 = Print nothing
   1 = Show recursive calls to DP
   2 = Show best trees found found each recursive call to DP
   3 = Show how many trees must be fully expanded and how many were timed
   4 = Show formulas that are being timed.
```

### Examples
```
spiral> DP( SPLNonTerminal("DCT2", 4) );   
DP called on SPLNonTerminal( "DCT2", 4 )
1 tree(s) to fully expand
  DP called on SPLNonTerminal( "DCT2", 2 )
  1 tree(s) to fully expand
  1 tree(s) timed at this level
  Best Trees:
     DCT2(2)     {RuleDCT2_0} ! 67
  DP called on SPLNonTerminal( "DCT4", 2 )
  1 tree(s) to fully expand
  1 tree(s) timed at this level
  Best Trees:
     DCT4(2)     {RuleDCT4_0} ! 84
1 tree(s) timed at this level
Best Trees:
   DCT2(4)     {RuleDCT2_2}
    |--DCT2(2)     {RuleDCT2_0}
    `--DCT4(2)     {RuleDCT4_0} ! 124
3 total trees timed
[ rec(
      ruletree := RuleTree(
          RuleDCT2_2,
          SPLNonTerminal( "DCT2", 4 ), [
          RuleTree(RuleDCT2_0, SPLNonTerminal( "DCT2", 2 )),
          RuleTree(RuleDCT4_0, SPLNonTerminal( "DCT4", 2 ))
        ] ),
      measured := 124 ) ]
```

```
spiral> DP( SPLNonTerminal("DCT4", 4), rec(nBest:=2), rec() );
DP called on SPLNonTerminal( "DCT4", 4 )
10 tree(s) to fully expand
  DP called on SPLNonTerminal( "DCT2", 4 )
  1 tree(s) to fully expand
    DP called on SPLNonTerminal( "DCT2", 2 )
    1 tree(s) to fully expand
    1 tree(s) timed at this level
    Best Trees:
       DCT2(2)     {RuleDCT2_0} ! 67
    DP called on SPLNonTerminal( "DCT4", 2 )
    1 tree(s) to fully expand
    1 tree(s) timed at this level
    Best Trees:
       DCT4(2)     {RuleDCT4_0} ! 83
  1 tree(s) timed at this level
  Best Trees:
     DCT2(4)     {RuleDCT2_2}
      |--DCT2(2)     {RuleDCT2_0}
      `--DCT4(2)     {RuleDCT4_0} ! 124
  DP called on SPLNonTerminal( "DST2", 2 )
  1 tree(s) to fully expand
  1 tree(s) timed at this level
  Best Trees:
     DST2(2)     {RuleDST2_0} ! 65
  DP called on SPLNonTerminal( "DST4", 2 )
  1 tree(s) to fully expand
  1 tree(s) timed at this level
  Best Trees:
     DST4(2)     {RuleDST4_0} ! 83
  DP called on SPLNonTerminal( "DCT3", 4 )
  1 tree(s) to fully expand
    DP called on SPLNonTerminal( "DCT3", 2 )
    1 tree(s) to fully expand
    1 tree(s) timed at this level
    Best Trees:
       DCT3(2)     {RuleDCT2_0 ^ T} ! 73
  1 tree(s) timed at this level
  Best Trees:
     DCT3(4)     {RuleDCT2_2 ^ T}
      |--DCT3(2)     {RuleDCT2_0 ^ T}
      `--DCT4(2)     {RuleDCT4_0} ! 135
  DP called on SPLNonTerminal( "DST3", 2 )
  1 tree(s) to fully expand
  1 tree(s) timed at this level
  Best Trees:
     DST3(2)     {RuleDST2_0 ^ T} ! 70
10 tree(s) timed at this level
Best Trees:
   DCT4(4)     {RuleDCT4_5 ^ T} ! 162
   DCT4(4)     {RuleDCT4_3}
    |--DCT2(2)     {RuleDCT2_0}
    `--DST2(2)     {RuleDST2_0} ! 170
18 total trees timed
[ rec(
      ruletree := RuleTree(RuleDCT4_5, "T", SPLNonTerminal( "DCT4", 4 )),
      measured := 162 ), rec(
      ruletree := RuleTree(
          RuleDCT4_3,
          SPLNonTerminal( "DCT4", 4 ), [
          RuleTree(RuleDCT2_0, SPLNonTerminal( "DCT2", 2 )),
          RuleTree(RuleDST2_0, SPLNonTerminal( "DST2", 2 ))
        ] ),
      measured := 170 ) ]
spiral> 
```

RandomSearch
------------
 
Usage: ```RandomSearch( <spl> [, <RandomSearch-options-record>, <SPL-options-record> ] )```

Note that if you wish to specify a RandomSearch-options-record or a
SPL-options-record, then you must specify both.  To leave one blank,
just pass "rec()" in its place.

RandomSearch generates random ruletrees and times them, keeping track of
the fastest one it has found so far.  Technically, RandomSearch() is just
a front in to STEER(), passing the correct options to only generate and
time random formulas.

Returns: The fastest implementation found as an "individual" which is a 
record consisting of fields for a ruletree, SPLOptions, the measured event
(usually runtime), and a few other fields not of importance.

### Options
```
spiral> PrintSpecRandomSearchOptionsRecord();
rec(
  numFormulas := <positive integer>
  seed        := <integer>
  timeLimit := false | <minutes>,
  localUnrolling := true | false,
  localUnrollingMin := <positive int>,
  localUnrollingMax := <positive int>,
  globalUnrolling := true | false,
  globalUnrollingMin := <positive int>,
  globalUnrollingMax := <positive int>,
  bestFound := "save" | "none",
  verbosity   := <non-negative integer>
);
```
Run 'MergeRandomSearchOptionsRecord( rec() );' to determine defaults.

numFormulas specifies how many random formulas to generate.  Note that is
possible that RandomSearch will not time this many formulas as it will not
time the exact same formula twice.

seed specifies a random seed for use with the random number generator.

### Verbosity Levels
```
   0 = Print nothing
   1 = Print final best
   2 = Print formulas being timed
```

### Example

```
spiral> RandomSearch( SPLNonTerminal("DFT",16), rec(numFormulas:=10), 
> rec(globalUnrolling:=8) );

Summary:
  Indiv 1: DFT(16)     {RuleDFT_1}
             |--DFT(4)     {RuleDFT_1 ^ T}
             |   |--DFT(2)     {RuleDFT_0}
             |   `--DFT(2)     {RuleDFT_0}
             `--DFT(4)     {RuleDFT_1}
                 |--DFT(2)     {RuleDFT_0}
                 `--DFT(2)     {RuleDFT_0} ! 2861
rec(
  IsIndiv := true,
  operations := IndivOps,
  ruletree := RuleTree(
      RuleDFT_1,
      SPLNonTerminal( "DFT", 16 ), [
      RuleTree(
        RuleDFT_1, "T",
        SPLNonTerminal( "DFT", 4 ), [
        RuleTree(RuleDFT_0, SPLNonTerminal( "DFT", 2 )),
        RuleTree(RuleDFT_0, SPLNonTerminal( "DFT", 2 ))
      ] ),
      RuleTree(
        RuleDFT_1,
        SPLNonTerminal( "DFT", 4 ), [
        RuleTree(RuleDFT_0, SPLNonTerminal( "DFT", 2 )),
        RuleTree(RuleDFT_0, SPLNonTerminal( "DFT", 2 ))
      ] )
    ] ),
  SPLOpts := rec(
      dataType := "complex",
      globalUnrolling := 8,
      language := "fortran",
      compflags := "'-O -fomit-frame-pointer -malign-double'" ),
  measured := 2861,
  fitness := 1/2861 )
spiral>
```

HillClimb
---------
Usage: ```HillClimb( <spl> [, <HillClimb-options-record>, <SPL-options-record> ] )```

Note that if you wish to specify a HillClimb-options-record or a
SPL-options-record, then you must specify both.  To leave one blank,
just pass "rec()" in its place.

HillClimb performs a hill climbing search.  It begins by generating a
random implementation and timing it.  Next, it applies a random mutation
to generate a new implementation which is then timed.  If this new
implementation is faster, then a random mutation is applied to the new
implementation and otherwise a random mutation is applied to the
original implementation.  Mutations are applied a specified number of
times, searching for a fast implementation.  Then the process is
restarted with a new random implementation.

Returns: The fastest implementation found as an "individual" which is a 
record consisting of fields for a ruletree, SPLOptions, the measured event
(usually runtime), and a few other fields not of importance.

### Options
```
spiral> PrintSpecHillClimbOptionsRecord();
rec(
  timeLimit := false | <minutes>,
  localUnrolling := true | false,
  localUnrollingMin := <positive int>,
  localUnrollingMax := <positive int>,
  globalUnrolling := true | false,
  globalUnrollingMin := <positive int>,
  globalUnrollingMax := <positive int>,
  bestFound := "save" | "none",
  numRestart  := <positive integer>
  quitRestart := <positive integer>
  numMutate   := <positive integer>
  quitMutate  := <positive integer>
  seed        := <integer>
  hashTable   := <hashTable>
  verbosity   := <non-negative integer>
);
```

Run ```MergeHillClimbOptionsRecord( rec() );``` to determine defaults.

numRestart specifies the maximum number of times HillClimb will restart
with a random implementation.

quitRestart specifies the maximum number of restarts without seeing an
improvement in the best found implementation.

numMutate specifies the maximum number of mutates HillClimb will perform
before restarting.

quitMutate specifies the maximum number of mutates (before restarting)
without seeing an improvement in the current implementation.

seed specifies a random seed for use with the random number generator.

HillClimb keeps track of all the formulas it has timed so far (to avoid
duplicating timings).  This information is kept in a hash table.  It is
possible to pass a hash table to HillClimb for it to use and so that you
can keep the resulting hash table when HillClimb finishes.  Note that
any entries in the hashTable will be assumed to have correct timings.
You can create a new hashTable for use with HillClimb by calling
HashTableHillClimb().

### Verbosity Levels
```
   0 = Print nothing
   1 = Print final best
   2 = Print restart count
   3 = Print restart best found so far
   4 = Print formulas being timed
```

### Example
```
spiral> HillClimb( Transform("DST2",8), rec(numRestart:=3), rec() );;
Restart 1
   10 mutations tried
   New Best Found:
      DST3(8)     {RuleDST2_2 ^ T}
       `--DCT3(8)     {RuleDCT2_2 ^ T}
           |--DCT3(4)     {RuleDCT2_2 ^ T}
           |   |--DCT3(2)     {RuleDCT2_0 ^ T}
           |   `--DCT4(2)     {RuleDCT4_0}
           `--DCT4(4)     {RuleDCT4_3 ^ T}
               |--DCT3(2)     {RuleDCT2_0 ^ T}
               `--DST3(2)     {RuleDST2_0 ^ T} ! 336
Restart 2
   15 mutations tried
   New Best Found:
      DST3(8)     {RuleDST2_3 ^ T}
       |--DST4(4)     {RuleDST4_1}
       |   `--DCT4(4)     {RuleDCT4_1}
       |       `--DCT2(4)     {RuleDCT2_2}
       |           |--DCT2(2)     {RuleDCT2_0}
       |           `--DCT4(2)     {RuleDCT4_0}
       `--DST3(4)     {RuleDST2_3 ^ T}
           |--DST4(2)     {RuleDST4_0}
           `--DST3(2)     {RuleDST2_0 ^ T} ! 307
Restart 3
   11 mutations tried
   No better finds than previous 307

7 total trees timed
Best Found Implementation:
DST3(8)     {RuleDST2_3 ^ T}
    |--DST4(4)     {RuleDST4_1}
    |   `--DCT4(4)     {RuleDCT4_1}
    |       `--DCT2(4)     {RuleDCT2_2}
    |           |--DCT2(2)     {RuleDCT2_0}
    |           `--DCT4(2)     {RuleDCT4_0}
    `--DST3(4)     {RuleDST2_3 ^ T}
        |--DST4(2)     {RuleDST4_0}
        `--DST3(2)     {RuleDST2_0 ^ T} ! 307
```


STEER (Split Tree Evolution for Efficient Runtimes)
---------------------------------------------------
 
Usage: ```STEER( <spl> [, <STEER-options-record>, <SPL-options-record> ] )```

Note that if you wish to specify a STEER-options-record or a
SPL-options-record, then you must specify both.  To leave one blank,
just pass "rec()" in its place.

STEER is a stochastic evolutionary search algorithm for finding fast
implementations.  STEER is very similar to a genetic algorithm.

Returns: The fastest implementation found as an "individual" which is a 
record consisting of fields for a ruletree, SPLOptions, the measured event
(usually runtime), and a few other fields not of importance.

### Options
```
spiral> PrintSpecSTEEROptionsRecord();
rec(
  timeLimit := false | <minutes>,
  localUnrolling := true | false,
  localUnrollingMin := <positive int>,
  localUnrollingMax := <positive int>,
  globalUnrolling := true | false,
  globalUnrollingMin := <positive int>,
  globalUnrollingMax := <positive int>,
  bestFound := "save" | "none",
  popSize  := <positive integer>
  numGens  := <positive integer>
  quitGen  := <positive integer>
  bestKept := <non-negative integer>
  crossed  := <non-negative integer>
  mutated  := <non-negative integer>
  injected := <non-negative integer>
  seed     := <integer>
  fitnessFun := ReciprocalFitness | MeasuredFitness | <user-specified-function>
  hashTable := <hashTable>,
  verbosity := <non-negative integer>
);
```

Run 'MergeSTEEROptionsRecord( rec() );' to determine defaults.

popSize specifies the size of the population.  That is, popSize formulas
are present in the population each generation.

numGens specifies the maximum number of generations that STEER will be
allowed to run.

quitGen specifies that after the given number of generations without finding
a faster formula than the current best, STEER should stop.

bestKept specifies the number of distinct fastest formulas to keep from
generation to generation.

crossed specifies the number of formulas to cross-over each generation.
Note that crossed/2 pairs of formulas are crossed-over.

mutated specifies the number of formulas to be mutated each generation.

injected specifies the number of new random formulas to be injected into
the population each generation after the first.

Note that it must be that popSize >= bestKept + crossed + mutated + injected.

seed specifies a random seed for use with the random number generator.

fitnessFun specifies a function used to calculate the fitness of a formula
given its measured event.  The higher the fitness, the better the formula.

STEER keeps track of all the formulas it has timed so far (to avoid
duplicating timings).  This information is kept in a hash table.
It is possible to pass a hash table to STEER for it to use and so that you
can keep the resulting hash table when STEER finishes.  Note that any
entries in the hashTable will be assumed to have correct timings.  You can
create a new hashTable for use with STEER by calling HashTableSTEER().

### Verbosity Levels
```
   0 = Print nothing
   1 = Print final best
   2 = Print generation count
   3 = Print population stats for each generation
   4 = Print formulas being timed
```

### Example

```
spiral> STEER(  SPLNonTerminal("DCT1",4),           
> rec( seed := 348, bestKept := 2, verbosity := 1 ), rec() );

Summary:
  Indiv 1: DCT1(4)     {RuleDCT1_3 ^ T}
             |--DCT1(2)     {RuleDCT1_3}
             |   |--DCT1(1)     {RuleDCT1_0}
             |   `--DCT1(1)     {RuleDCT1_0}
             `--DCT1(2)     {RuleDCT1_3}
                 |--DCT1(1)     {RuleDCT1_0}
                 `--DCT1(1)     {RuleDCT1_0} ! 149
  Indiv 2: DCT1(4)     {RuleDCT1_3}
             |--DCT1(2)     {RuleDCT1_3 ^ T}
             |   |--DCT1(1)     {RuleDCT1_0}
             |   `--DCT1(1)     {RuleDCT1_0}
             `--DCT1(2)     {RuleDCT1_3 ^ T}
                 |--DCT1(1)     {RuleDCT1_0}
                 `--DCT1(1)     {RuleDCT1_0} ! 153

rec(
  IsIndiv := true,
  operations := IndivOps,
  ruletree := RuleTree(
      RuleDCT1_3, "T",
      SPLNonTerminal( "DCT1", 4 ), [
      RuleTree(
        RuleDCT1_3,
        SPLNonTerminal( "DCT1", 2 ), [
        RuleTree(RuleDCT1_0, SPLNonTerminal( "DCT1", 1 )),
        RuleTree(RuleDCT1_0, SPLNonTerminal( "DCT1", 1 ))
      ] ),
      RuleTree(
        RuleDCT1_3,
        SPLNonTerminal( "DCT1", 2 ), [
        RuleTree(RuleDCT1_0, SPLNonTerminal( "DCT1", 1 )),
        RuleTree(RuleDCT1_0, SPLNonTerminal( "DCT1", 1 ))
      ] )
    ] ),
  SPLOpts := rec(
      dataType := "real",
      globalUnrolling := 32,
      language := "fortran",
      compflags := "'-O -fomit-frame-pointer -malign-double'" ),
  measured := 149,
  fitness := 1/149 )
spiral> 
```


TimedSearch
-----------
 
Usage: ```TimedSearch( <spl> [, <TimedSearch-options-record>, <SPL-options-record> ] )```

Note that if you wish to specify a TimedSearch-options-record or a
SPL-options-record, then you must specify both.  To leave one blank,
just pass "rec()" in its place.

TimedSearch is a meta-search algorithm; that is, it calls other search
algorithms to do the real search.  It runs for a specified length of time,
limiting the called search algorithms to certain amounts of time.  The
idea is that this algorithm can be used to find the best ruletree possible
in say 30 minutes (or any specified length of time).

Returns: Best Found Implementation (this is a record containing a ruletree,
the SPL Options used, and the measured amount of time).  Note that
TimedSearch does a BestFoundLookup and returns the resulting implementation;
so, it is possible for TimedSearch to return a ruletree that none of the
called search algorithms timed during the execution of TimedSearch.

### Options:
```
spiral> PrintSpecTimedSearchOptionsRecord();
rec(
  timeLimit := <minutes>,
  searches  := [ "<searchMethod1>", <SearchOpts1>, <timeLimit1>,
                 ...
                 "<searchMethodN>", <SearchOptsN>, <timeLimitN> ]
  verbosity := <non-negative integer>
);
```
Run 'MergeTimedSearchOptionsRecord( rec() );' to determine defaults.

timeLimit specifies the overall time limit for the entire search.  This
can not be set to false as in the other search algorithms.

Setting searches determines which search algorithms are called from
TimedSearch, with which search options, and for how long.  Each search
method is called in turn, passed its respective search options.  Each
search method has its time limit set to be the minimum of timeLimitK and
of the remaining global time.

searchMethodK must be a string and the name of a valid search algorithm.
searchOptsK must be a valid search options record for searchMethodK.
timeLimitK must be an integer representing the maximum time in minutes,
   or false if searchMethodK is to be allowed to run up to the maximum
   global amount of remaining time.

The default is to run RandomSearch on 10 formulas, then to run DP, next
STEER, and then to run 4-best DP over localUnrolling and finally STEER
over localUnrolling with a larger population (see config.g).

### Verbosity Levels:
```
   0 = Print nothing, tell search algorithms to print nothing
   1 = Print nothing
   2 = Print final best
   3 = Print search algorithms being called
```



Hash Tables
-----------

Hash tables are efficient ways to store certain types of data.  They are
used in several places in the search module.  For example, the
BestFoundTable is implemented as a HashTable with particular wrapper
functions.  Also, DP and STEER use HashTables to store partial results.
These hash tables can be saved to files and later reused to avoid
duplicating work already done by the search algorithms.

In particular, DP uses a HashTable to store the nBest ruletrees that it
has found for a given spl.  STEER uses a HashTable to store entire
implementations to avoid re-running the same implementation multiple
times.

The most common reason why you would want to use HashTables is to save
DP's partial results.  This is particularly useful, if, for example, you
run DP on a transform of a particular size, but later may want to run DP
on a larger size of that transform.  By saving off DP's HashTable and
then later reusing it, DP can avoid duplicating its earlier work.  An
example is shown below.


### HashSave()

Usage: ```HashSave( HashTable, filename )```

Saves the given HashTable to the specified filename (given as a string).
This allows later retrieval of the HashTable.


### HashRead()

Usage: ```HashRead( filename )```

Returns the HashTable that was stored in the specified file (filename
should be passed as a string).


### HashTable Creation


- **HashTableDP()** creates a HashTable for use with DP.
- **HashTableSTEER()** creates a HashTable for use with STEER.


### Example

```
spiral> myHashTable := HashTableDP();
HashTable
spiral> DP( Transform("DFT",32), rec(hashTable:=myHashTable), rec() );
...
spiral> HashSave( myHashTable, "DP_DFT32.hash" );
spiral> quit;
$ 
...
spiral> myHashTable := HashRead( "DP_DFT32.hash" );
HashTable
spiral> DP( Transform("DFT",1024), rec(hashTable:=myHashTable), rec() );
...
```

BestFoundTable
--------------

The BestFoundTable keeps track of the best found implementations for the
different SPLs.  The search algorithms will save the best
implementations that they find to the BestFoundTable.  Currently, none
of the search algorithms use the table to guide their search.

You can interact with the BestFoundTable using the following functions:


### BestFoundLookup()
 

Usage: ```BestFoundLookup( <spl> [, <SPL-options-record>] )```

Looks up the best found implementation for the given spl.

Returns: a list of BestFoundImpl records consisting of the ruletree, the
actual full SPLOptions, and the measured time.

If no SPL-options-record is passed, then BestFoundLookup returns a list
of the (equally) fastest implementations across all possible SPL Options
(such as language or dataType).  That is, if there is a BestFound
Implementation for the given spl in both C and Fortran, it will only
return the one that is fastest, unless they are equally fast in which
case both will be returned.

If an SPL-options-record IS passed, then BestFoundLookup returns a list 
of the (equally) fastest implementations across only those
implementations having the same values for the SPL Options listed in the
variable BestFoundDifferFields.  By default, BestFoundDifferFields
includes the fields "dataType" and "language".

So, if you want to get the BestFound Implementation for a DFT of size 16
implemented in C (but irrespective of the used dataType), try:
   BestFoundLookup( Transform("DFT",16), rec(language:="c") );


### BestFoundSave()
 

Usage: ```BestFoundSave( filename )```

Saves the BestFoundTable to the specified filename (given as a string).
This allows later retrieval of the Table.


### BestFoundRead()
 

Usage: ```BestFoundRead( filename )```

Reads the BestFoundTable in the specified filename (given as a string).
This overwrites (without saving) the current BestFoundTable.  In case of
error in reading the file, the current BestFoundTable is not overwritten.
