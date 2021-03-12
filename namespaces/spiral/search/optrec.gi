
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# Option Records
# ==============
# MP, BS, from 01/25/01, GAPv3.4.4


# general default search options
SEARCH_DEFAULTS :=
    rec(
		timeLimit          := false,
		globalUnrolling    := false,
		globalUnrollingMin := 8,
		globalUnrollingMax := 64,
		timeBaseCases      := true
    );

# default DP search options
DP_DEFAULTS :=
    rec(
        nBest     := 1,
        optimize  := "minimize",
        verbosity := 3,
	    DPVec     := false,
	    DPVecVlen := 4
    );


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
   Print("  globalUnrolling := true | false,\n");
   Print("  globalUnrollingMin := <positive int>,\n"); 
   Print("  globalUnrollingMax := <positive int>,\n");
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
