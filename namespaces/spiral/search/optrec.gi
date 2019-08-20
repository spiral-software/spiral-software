
# Copyright (c) 2018-2019, Carnegie Mellon University
# See LICENSE for details


# Option Records
# ==============
# MP, BS, from 01/25/01, GAPv3.4.4


#F Search Options Record
#F ---------------------

#F Search options records keep search options that are not specific to any
#F particular search algorithm.  These are options that cause the search
#F algorithms to search over different implementations instead of just how
#F the search algorithm progresses.
#F 
#F These options get incorporated into a search algorithm's options record.
#F 


#F PrintSpecSearchOptionsRecord()
#F   prints the specification for the general search options
#F   only to be called by specific search algorithm's PrintSpec
#F 

PrintSpecSearchOptionsRecord := function()
   Print("  timeLimit := false | <minutes>,\n");
   Print("  localUnrolling := true | false,\n");
   Print("  localUnrollingMin := <positive int>,\n");
   Print("  localUnrollingMax := <positive int>,\n");
   Print("  globalUnrolling := true | false,\n");
   Print("  globalUnrollingMin := <positive int>,\n"); 
   Print("  globalUnrollingMax := <positive int>,\n"); 
   Print("  bestFound := \"save\" | \"none\",\n");
end;


#F CheckSearchOptionsRecord( <search-options-record> )
#F   checks whether <search-options-record> is a valid search options record
#F

CheckSearchOptionsRecord := function( R )
   local r;

   if not IsRec(R) then
    Error("<R> must be an search options record");
  fi;

  # check fields
  for r in RecFields(R) do
     if r = "timeLimit" then
        if not ( R.(r) = false or ( IsInt(R.(r)) and R.(r) > 0 ) ) then
       Error( "timeLimit must be either false or a positive integer" );
    fi;
     elif r = "localUnrolling" then
        if not IsBool(R.(r)) then
       Error( "Search option localUnrolling must be true or false" );
    fi;
     elif r = "localUnrollingMin" then
        if not IsInt(R.(r)) then
       Error( "localUnrollingMin is not an integer" );
    fi;
     elif r = "localUnrollingMax" then
        if not IsInt(R.(r)) then
       Error( "localUnrollingMax is not an integer" );
    fi;
     elif r = "globalUnrolling" then
        if not IsBool(R.(r)) then
       Error( "Search option globalUnrolling must be true or false" );
    fi;
     elif r = "globalUnrollingMin" then
        if not IsInt(R.(r)) then
       Error( "globalUnrollingMin is not an integer" );
    fi;
     elif r = "globalUnrollingMax" then
        if not IsInt(R.(r)) then
       Error( "globalUnrollingMax is not an integer" );
    fi;
     elif r = "bestFound" then
        if not IsString(R.(r)) and R.(r) in ["save","none"] then
       Error( "bestFound must be either \"save\" or \"none\"" );
    fi;
     elif r = "timeBaseCases" then
        if not IsBool(R.(r)) then
       Error( "Search option timeBaseCases must be true or false" );
    fi;
     fi;
  od;

  return true;
end;


#F MergeSearchOptionsRecord( <search-options-record> )
#F   merges <search-options-record> with the default values
#F

MergeSearchOptionsRecord := function( R )
   local OR, r;

   CheckSearchOptionsRecord(R);

   OR := ShallowCopy(R);
   for r in RecFields(SEARCH_DEFAULTS) do
      if not IsBound( OR.(r) ) then
         OR.(r) := SEARCH_DEFAULTS.(r);
      fi;
   od;

   if OR.localUnrolling = true and OR.globalUnrolling = true then
      Error( "Can not search over both local and global Unrolling" );
   fi;
   if OR.localUnrolling = true and
      OR.localUnrollingMin > OR.localUnrollingMax then
      Error( "localUnrollingMin > localUnrollingMax" );
   fi;
   if OR.globalUnrolling = true and
      OR.globalUnrollingMin > OR.globalUnrollingMax then
      Error( "globalUnrollingMin > globalUnrollingMax" );
   fi;

   return OR;
end;



#F DP Options Record
#F -----------------

#F The DP Options Records maintain options for the Dynamic Programming search
#F algorithm.  This is a combination of both search options specific to DP
#F as well as general search options.

#F PrintSpecDPOptionsRecord()
#F   prints the specification for the DP search options
#F 

PrintSpecDPOptionsRecord := function()
   Print("rec(\n");
   PrintSpecSearchOptionsRecord();
   Print("  nBest := <positive integer>,\n");
   Print("  optimize := \"minimize\" | \"maximize\",\n");
   Print("  hashTable := <hashTable>,\n");
   Print("  verbosity := <non-negative integer>\n");
   Print("  saveMemory := <boolean>\n");
   Print(");\n");
end;


#F CheckDPOptionsRecord( <DP-options-record> )
#F   checks to see if DP-options-record is valid
#F

CheckDPOptionsRecord := function( R )
   local r;

   CheckSearchOptionsRecord(R);
   for r in RecFields(R) do
      if r = "nBest" then
         if not ( IsInt(R.(r)) and R.(r) > 0 ) then
        Error( "nBest must be a positive integer" );
     fi;
      elif r = "verbosity" then
         if not ( IsInt(R.(r)) and R.(r) >= 0 ) then
        Error( "verbosity must be a non-negative integer" );
     fi;
      elif r = "optimize" then
         if not R.(r) in ["minimize","maximize"] then
        Error( "optimize must be to minimize or maximize" );
     fi;
      elif r = "hashTable" then
         if not IsHashTable( R.(r) ) then
        Error( "hashTable must be a valid hashTable" );
         fi;
      elif r = "hashTableBases" then
         if not IsHashTable( R.(r) ) then
        Error( "hashTableBases must be a valid hashTable" );
         fi;
      elif r = "breakdownRules" then
         if not IsRec( R.(r) ) then
        Error( "breakdownRules must be a record" );
         fi;
      elif r = "saveMemory" then
         if not IsBool(R.(r)) then
            Error("saveMemory must be boolean");
         fi;
      elif not r in RecFields(SEARCH_DEFAULTS) and  r <> "DPVec" and r <> "measureFunction" and r <> "wrap" 
      and not IsSystemRecField(r) then
         Error( "Unknown DPOptionsRecord field <r>" );
      fi;
   od;
end;


#F MergeDPOptionsRecord( <DP-options-record> )
#F   merges <DP-options-record> with the defaults
#F

MergeDPOptionsRecord := function( R )
   local OR, r;

   CheckDPOptionsRecord(R);
   OR := MergeSearchOptionsRecord(R);
   for r in RecFields(DP_DEFAULTS) do
      if not IsBound( OR.(r) ) then
         OR.(r) := DP_DEFAULTS.(r);
      fi;
   od;

   return OR;
end;


#F Exhaustive Search Options Record
#F --------------------------------

#F The Exhaustive Search Options Records maintain options for exhaustive search.
#F

#F PrintSpecExhaustiveSearchOptionsRecord()
#F   prints the specification for the ExhaustiveSearch search options
#F 

PrintSpecExhaustiveSearchOptionsRecord := function()
   Print("rec(\n");
   Print("  timeLimit := false | <minutes>,\n");
   Print("  bestFound := \"save\" | \"none\",\n");
   Print("  verbosity := <non-negative integer>\n");
   Print(");\n");
end;


#F CheckExhaustiveSearchOptionsRecord( <ExhaustiveSearch-options-record> )
#F   checks to see if ExhaustiveSearch-options-record is valid
#F

CheckExhaustiveSearchOptionsRecord := function( R )
   local r;

   CheckSearchOptionsRecord(R);
   for r in RecFields(R) do
      if r = "verbosity" then
         if not (IsInt(R.(r)) and R.(r) >= 0) then
        Error( "verbosity must be a non-negative integer" );
     fi;
      elif not r in [ "bestFound", "timeLimit" ] and not IsSystemRecField(r) then
         Error( "Unknown ExhaustiveSearchOptionsRecord field <r>" );
      fi;
   od;
end;


#F MergeExhaustiveSearchOptionsRecord( <ExhaustiveSearch-options-record> )
#F   merges <ExhaustiveSearch-options-record> with the defaults
#F

MergeExhaustiveSearchOptionsRecord := function( R )
   local OR, r;

   CheckExhaustiveSearchOptionsRecord(R);
   OR := ShallowCopy(R);
   for r in RecFields(EXHAUSTIVE_SEARCH_DEFAULTS) do
      if not IsBound( OR.(r) ) then
         OR.(r) := EXHAUSTIVE_SEARCH_DEFAULTS.(r);
      fi;
   od;
   if not IsBound( OR.bestFound ) then
      OR.bestFound := SEARCH_DEFAULTS.bestFound;
   fi;
   if not IsBound( OR.timeLimit ) then
      OR.timeLimit := SEARCH_DEFAULTS.timeLimit;
   fi;

   return OR;
end;




#F Timed Search Options Record
#F ---------------------------

#F The Timed Search Options Records maintain options for timed search.
#F

#F PrintSpecTimedSearchOptionsRecord()
#F   prints the specification for the TimedSearch search options
#F 

PrintSpecTimedSearchOptionsRecord := function()
   Print("rec(\n");
   Print("  timeLimit := <minutes>,\n");
   Print("  searches  := ",
         "[ \"<searchMethod1>\", <SearchOpts1>, <timeLimit1>,\n");
   Print("                 ...\n" );
   Print("                 ",
         "\"<searchMethodN>\", <SearchOptsN>, <timeLimitN> ]\n");
   Print("  verbosity := <non-negative integer>\n");
   Print(");\n");
end;


#F CheckTimedSearchOptionsRecord( <TimedSearch-options-record> )
#F   checks to see if TimedSearch-options-record is valid
#F

CheckTimedSearchOptionsRecord := function( R )
   local r, i;

   for r in RecFields(R) do
      if r = "timeLimit" then
         if not ( IsInt(R.(r)) and R.(r) > 0 ) then
        Error( "timeLimit must be a positive integer" );
     fi;
      elif r = "searches" then
         if not IsList( R.(r) ) then
        Error( "field 'searches' must be a list" );
     elif Length(R.(r)) = 0 then
        Error( "field 'searches' must be a non-empty list" );
     elif not ( Length(R.(r)) mod 3 ) = 0 then
        Error( "field 'searches' must be of the form:\n",
           "  [ \"<searchMethod1>\", <SearchOpts1>, <timeLimit1>,\n",
           "    ...\n",
           "    \"<searchMethodN>\", <SearchOptsN>, <timeLimitN> ]\n" );
     else
        for i in [1,4..(Length(R.(r))-2)]
        do
           if not R.(r)[i] in [ "DP", 
                                "ExhaustiveSearch", "TimedSearch" ] then
          Error( "field 'searches' must be of the form:\n",
           "  [ \"<searchMethod1>\", <SearchOpts1>, <timeLimit1>,\n",
           "    ...\n",
           "    \"<searchMethodN>\", <SearchOptsN>, <timeLimitN> ]\n",
               "searchMethod ", R.(r)[i], " not one of ",
           "\"DP\",  \n",
            " or \"ExhaustiveSearch\"" );
           elif not (    ( IsInt( R.(r)[i+2] ) and R.(r)[i+2] > 0 )
              or R.(r)[i+2] = false ) then
          Error( "field 'searches' must be of the form:\n",
           "  [ \"<searchMethod1>\", <SearchOpts1>, <timeLimit1>,\n",
           "    ...\n",
           "    \"<searchMethodN>\", <SearchOptsN>, <timeLimitN> ]\n",
               "timeLimit must be either false or a positive integer" );
           else
              if R.(r)[i] = "DP" then
             CheckDPOptionsRecord(R.(r)[i+1]);
          elif R.(r)[i] = "ExhaustiveSearch" then
             CheckExhaustiveSearchOptionsRecord(R.(r)[i+1]);
          elif R.(r)[i] = "TimedSearch" then
             CheckTimedSearchOptionsRecord(R.(r)[i+1]);
              fi;
           fi;
        od;
     fi;
      elif r = "verbosity" then
         if not (IsInt(R.(r)) and R.(r) >= 0) then
        Error( "verbosity must be a non-negative integer" );
     fi;
      elif not IsSystemRecField(r) then 
         Error( "Unknown TimedSearchOptionsRecord field <r>" );
      fi;
   od;
end;


#F MergeTimedSearchOptionsRecord( <TimedSearch-options-record> )
#F   merges <TimedSearch-options-record> with the defaults
#F

MergeTimedSearchOptionsRecord := function( R )
   local OR, r;

   CheckTimedSearchOptionsRecord(R);
   OR := ShallowCopy(R);
   for r in RecFields(TIMED_SEARCH_DEFAULTS) do
      if not IsBound( OR.(r) ) then
         OR.(r) := TIMED_SEARCH_DEFAULTS.(r);
      fi;
   od;

   return OR;
end;



# Implement Options Record
# ------------------------

#F The Implement Options Records maintain options for timed search.
#F

# Utility for Search algorithms to check dataType in SPLOpts
# ----------------------------------------------------------

#F SearchCheckDataType( <spl>, <SPL-options-record> )
#F   Checks to see that the dataType is set properly for the given spl.
#F   Sets it if no default was given.
#F

SearchCheckDataType := function( spl, SPLOpts )
   if not IsBound( SPLOpts.dataType ) then
      Error( "SPLOpts.dataType not bound" );
   fi;

   if SPLOpts.dataType = "no default" then
      if IsRealSPL(spl) then
         SPLOpts.dataType := "real";
      else
         SPLOpts.dataType := "complex";
      fi;
   elif SPLOpts.dataType in ["real","complex"] then
       ;
      # YEVGEN: This should not be too ambitious with error checking
      # we currently use real datatype for complex transforms (RC operator is applied
      # to get a real formula).

      #if (not IsRealSPL(spl)) and SPLOpts.dataType = "real" then
      #   Error("Real data type specified but spl is complex");
      #fi;
   else
      Error("SPLOpts.dataType is not \"real\" or \"complex\"");
   fi;
end;
