
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F HashIndex and HashEqual Functions
#F ---------------------------------
#F

#F HashLookup( hashTable, key )
#F    returns the data associated with key in hashTable.
#F    returns false if key is not found in hashTable.
#F
HashLookup := function( hashTable, key )
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

      if pos = false then
	  return false;
      else
 #      # catch the case of hash-identical but not identical spls
#       elif IsSPL(key) and not IsIdenticalSPL(entry[pos].key, key) then
#             chash := Copy(entry[pos].data); 
#             if(IsList(chash)) then
#                for index in [1..Length(chash)] do
#                   chash[index].ruletree := 
# 		  # NOTE:
#                   spiral.formgen.ApplyRuleTreeSPL(entry[pos].data[index].ruletree, key);
#                od;
#             else
# 		# NOTE:
#                chash.ruletree := 
#                spiral.formgen.ApplyRuleTreeSPL(entry[pos].data.ruletree, key);
#             fi; 
#             return chash;
#       else
      return entry[pos].data;
      fi;
   else
      return false;
   fi;
end;

MultiHashLookup := function(hashes, key)
    local h, lkup;
    for h in hashes do
        lkup := HashLookup(h, key);
	if lkup<>false then return lkup; fi;
    od;
    return false;
end;

Declare(HashIndexSPLRec);

HashIndexParams := function(param, index)
     local p;
     # hash symbolic Value(type, val) simply as <val>
     if IsValue(param) then param := param.v; fi;

     if IsFFE(param) then
     index := index + 1 + When(param <> GF(2).zero, 0, 1);
     elif IsInt(param) then
	 index := index * (param+1);
     elif IsString(param) then
	 index := index * (Length(param)+1);
      
     elif IsFloat(param) then
	 index := index + 1;         
     elif IsRat(param) then
	 index := index + 1;         
     elif IsFloat(param) then
	 index := index + 1;         
      ## NOTE: How to handle bools???
     elif IsBool(param) then
	 index := index * When(param, 2,3);  

     elif IsCyc(param) then
	 index := index + 1;
     elif IsList(param) then
	 for p in param do 
	     index := HashIndexParams(p, index);
	 od;
     elif IsMat(param) then
	 index := index * Product(DimensionsMat(param));
     elif IsPolynomial(param) then
	 index := index * LaurentDegree(param);

      # MP: now we have transforms as parameter and thus need to handle this case
      # by calling the function recursively   
     elif IsSPL(param) then
	 index := index + HashIndexSPLRec(param);

     elif IsClass(ObjId(param)) then
         index := index + BagAddr(ObjId(param));
     fi;
     return index;
end;

#F HashIndexSPLRec( <spl> )
#F   Does the actual dirty work for HashIndexSPL recursively
#F   Do not call directly
#F

HashIndexSPLRec := function( spl )
   local type, index, param, params;

   if not IsSPL(spl) then
      Error( "<spl> must be a valid SPL" );
   fi;

   if IsNonTerminal(spl) and IsBound(spl.params) then
      index := BagAddr(ObjId(spl));
      params := When(IsBound(spl.HashId), spl.HashId(), spl.params);
      if IsInt( params ) then
	  index := index * (params+1);
      else 
	  params := When(not IsList(params), [params], params);
	  for param in params do
	      index := HashIndexParams(param, index);
	  od;
      fi;
   elif IsBound(spl.children) then
       index := Product( spl.children(), a->HashIndexSPLRec(a) );
   elif IsBound(spl.element) and IsList(spl.element) then
       index := Length(spl.element);
   else
       index := BagAddr(ObjId(spl));
   fi;

   return index;
end;


#F HashIndexSPL( <spl>, <hashSize> )
#F    one possible method of returning a hash index (probably not that good).
#F    takes a SPL and returns a hash index.
#F
HashIndexSPL := function( spl, hashSize )
   if not IsSPL(spl) then
      Error( "<spl> must be a valid SPL" );
   fi;

   if IsBound(spl.hashIndex) then
      return ( (spl.hashIndex mod hashSize) + 1 );
   fi;

   spl.hashIndex := HashIndexSPLRec(spl);
   return ( (spl.hashIndex mod hashSize) + 1 );
end;


#F HashIndexRuleTree( <ruletree>, <hashSize> )
#F    one possible method of returning a hash index (probably not that good).
#F    takes a RuleTree and returns a hash index.
#F

HashIndexRuleTree := function( tree, hashSize )
   local index,
         child;

   index := BagAddr(tree.rule);
   if tree.transposed then
      index := index*3;
   else
      index := index*7;
   fi;
   if IsBound(tree.splOptions) and tree.splOptions="unrolled" then
      index := index*11;
   else
      index := index*13;
   fi;
   index := ((index+1) * (HashIndexSPL(tree.node, hashSize)+1))
            mod hashSize;
   for child in tree.children do
      index := ((index+1) * (HashIndexRuleTree(child, hashSize)+1))
               mod hashSize;
   od;
   return index+1;
end;


#F HashTableSPL() - create hashtable with SPLs as keys
HashTableSPL := function()
   return HashTable( HashIndexSPL, IsHashIdenticalSPL, "HashTableSPL()" );
end;
