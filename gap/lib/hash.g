# -*- mode: shell-script -*-
GlobalPackage(spiral.hash);

# Copyright (c) 2018-2021, Carnegie Mellon University
# All rights reserved.
# 
# See LICENSE file for full information

#F Hash Table Implementation
#F =========================
#F

# Default size of a created hash table
DEFAULT_HASH_SIZE := 32749;

HashOps := OperationsRecord( "HashOps" );


# HashSimplePrint( <hashtable> )
#    default way to print out a hash table by simplying saying "HashTable"
#

HashSimplePrint := function( hashTable )
   Print( "HashTable" );
end;


# HashRecPrint( <hashtable> )
#    print out a hash table as one big record
#

HashRecPrint := function( hashTable )
   Print( "rec(\n",
	  "   IsHashTable := true,\n",
	  "   operations  := HashOps,\n",
	  "   indexFun    := ", hashTable.indexFun, ",\n",
	  "   equalFun    := ", hashTable.equalFun, ",\n",
	  "   size        := ", hashTable.size, ",\n",
	  "   entries     := ", hashTable.entries, "\n",
	  ")\n" );
end;


# HashSavePrint( <hashtable> )
#    method for printing a hash table used by HashSave()
#    creates a sequence of operations that reconstructs the hash table
#    each hash table entry is on a seperate line
#

HashSavePrint := function( hashTable )
   local item, entry, l;
   Print( "ImportAll(spiral);\n" );
   Print( "savedHashTable := " );
   if IsString(hashTable.creator) then
      Print( hashTable.creator );
   else
      for l in hashTable.creator do
         Print(l);
      od;
   fi;
   Print( ";\n\n" );

   for entry in Filtered( hashTable.entries, a->true ) do
      for item in entry do
	 Print( "HashAdd(savedHashTable, ", item.key, ", ", item.data, ");\n" );
      od;
   od;
end;


# Setup default hash table printing method

HashOps.Print := HashSimplePrint;


#F IsHashTable( <obj> )
#F    tests whether <obj> is a HashTable.
#F

IsHashTable := function( hashTable )
   return IsRec(hashTable) and IsBound(hashTable.IsHashTable)
                           and hashTable.IsHashTable;
end;


#F HashTable( <hashIndexFun> [, <equalFun> ] [, <hashSize> ] [, <creator> ] )
#F    Returns a HashTable that uses the speicified hashing function.
#F    The hashIndexFun should be a function of two arguments:
#F        The first argument should be the key and
#F        the second argument should be the size of the hashTable;
#F    Optionally, may specify an equality function to test equality of keys.
#F    Optionally, may specify the hash table size.
#F    Optionally, may specify a string defining the creator.
#F

HashTable := function( arg )
   local hashIndexFun, hashSize, equalFun, creator,
	 usage,
         i;

   usage := ConcatenationString(
            "usage: HashTable( <hashIndexFun> [, <equalFun> ] \n",
            "                  [, <hashSize> ] [, <creator> ] )" );
   # decode arg
   if Length(arg) < 1  or Length(arg) > 4 then
      Error( usage );
   fi;
   hashIndexFun := arg[1];
   if not IsFunc(hashIndexFun) then
      Error( usage );
   fi;
   hashSize := DEFAULT_HASH_SIZE;
   equalFun := function(a,b) return a=b; end;
   creator  := [ "HashTable(", hashIndexFun, ")" ];
   for i in [2..Length(arg)] do
      if IsInt(arg[i]) then
	 if arg[i] < 1 then
	    Error( "<hashSize> must be greater than 0" );
	 fi;
	 hashSize := arg[i];
      elif IsFunc(arg[i]) then
         equalFun := arg[i];
      elif IsString(arg[i]) then
         creator := arg[i];
      else
	 Error( usage );
      fi;
   od;

   return rec (
      IsHashTable  := true,
      operations   := HashOps,
      indexFun     := hashIndexFun,
      equalFun     := equalFun,
      size         := hashSize,
      creator      := creator,
      entries      := []
   );
end;


#F HashLookupDefault( hashTable, key )
#F    returns the data associated with key in hashTable.
#F    returns false if key is not found in hashTable.
#F
HashLookupDefault := function( hashTable, key )
   local index, entry, pos, chash, temp;
   if not IsHashTable( hashTable ) then
      Error( "<hashTable> is not a valid HashTable" );
   fi;
   index := hashTable.indexFun( key, hashTable.size );
   if not ( index > 0 and index <= hashTable.size ) then
      Error( "Did not obtain a valid hash index for key" );
   fi;

   if IsBound( hashTable.entries[index] ) then
      entry := hashTable.entries[index];
      pos := PositionProperty( entry, a -> hashTable.equalFun(a.key, key) );
      if pos = false then return false;
      else return entry[pos].data;
      fi;
   else
      return false;
   fi;
end;
HashLookupUID := function( hashTable, key )
   local index, entry, pos, chash, temp;
   if not IsHashTable( hashTable ) then
      Error( "<hashTable> is not a valid HashTable" );
   fi;
   index := hashTable.indexFun( key, hashTable.size );
   if not ( index > 0 and index <= hashTable.size ) then
      Error( "Did not obtain a valid hash index for key" );
   fi;

   if IsBound( hashTable.entries[index] ) then
      entry := hashTable.entries[index];
      pos := PositionProperty( entry, a -> hashTable.equalFun(a.key, key) );
      if pos = false then return false;
      else return [index,pos];
      fi;
   else
      return false;
   fi;
end;

HashLookup := HashLookupDefault;

#F HashAdd( hashTable, key, data )
#F    adds the key-data pair to the hashTable.
#F    Note -- a copy is not made of the key nor the data.
#F    returns the (unique) position in the hash table

HashAdd := function( hashTable, key, data )
   local index, bucket, pos;

   if not IsHashTable( hashTable ) then
      Error( "<hashTable> is not a valid HashTable" );
   fi;

   index := hashTable.indexFun( key, hashTable.size );

   if not ( index > 0 and index <= hashTable.size ) then
      Error( "Did not obtain a valid hash index for key" );
   fi;

   if IsBound( hashTable.entries[index] ) then
       bucket := hashTable.entries[index];
       # now search within a bucket
       pos := PositionProperty( bucket, a -> hashTable.equalFun(a.key, key) );
       if pos = false then # not found
           Add(bucket, rec(key := key, data := data));
           return [index, Length(bucket)];
       else # found in a bucket, replace the data
           bucket[pos].data := data; 
           return [index,pos];
       fi;
   else
      hashTable.entries[index] := [ rec(key := key, data := data) ];
      return [index, 1];
   fi;
end;


#F HashDelete( hashTable, key )
#F    deletes any key-data pair in hashTable matching the given key
#F

HashDelete := function( hashTable, key )
   local index;

   if not IsHashTable( hashTable ) then
      Error( "<hashTable> is not a valid HashTable" );
   fi;

   index := hashTable.indexFun( key, hashTable.size );

   if not ( index > 0 and index <= hashTable.size ) then
      Error( "Did not obtain a valid hash index for key" );
   fi;

   if IsBound( hashTable.entries[index] ) then
      hashTable.entries[index] :=
         Filtered( hashTable.entries[index],
	           entry -> not hashTable.equalFun(entry.key, key) );
   fi;
end;


#F HashSave( hashTable, filename )
#F    saves hashTable to filename for later recall
#F    use HashRead to read hashTables.
#F

HashSave := function( hashTable, filename )
   if not IsHashTable( hashTable ) then
      Error( "<hashTable> is not a valid HashTable" );
   fi;
   PrintTo( filename, HashSavePrint(hashTable) );
end;


#F HashRead( filename )
#F    returns hashTable read from filename.
#F    use HashSave to save hashTables.

HashRead := function( filename )
   local savedHashTable;
   savedHashTable := "Problems Reading HashTable";
   Read( filename );
   return savedHashTable;
end;

#F HashWalk(hashTable, func)
#F    lists <hashTable> items through <func> callback.
#F    <func> must have (<key>, <data>) signature.
#F

HashWalk := function( hashTable, func )
    local i, b;

    for i in [1..Length(hashTable.entries)] do
        if IsBound(hashTable.entries[i]) then
            for b in hashTable.entries[i] do
                func(b.key, b.data);
            od;
        fi;
    od;
end;

