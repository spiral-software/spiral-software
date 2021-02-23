
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


fId.canTensorSplit := (self, div) >> self.domain() mod div = 0;
fId.tensorSplit    := (self, div) >> [fId(div), fId(self.size/div)];
fBase.canTensorSplit := (self, div) >> self.range() mod div = 0;
fBase.tensorSplit := (self, div) >> [ fBase(div,        idiv(self.params[2], self.range()/div)), 
                                      fBase(self.range()/div, imod(self.params[2], self.range()/div)) ]; 


CanTensorSplit := (what, div) -> When(not IsRec(what) or not IsBound(what.canTensorSplit), false, 
                                      what.canTensorSplit(div));

TensorSplit    := (what, div) -> what.tensorSplit(div);


try_split := function(i, flist, fdim, N)
    local ffrun, next, len, split;
    len := Length(flist);
    ffrun := fdim(flist[i]);
    while i < len and ffrun < N do
        next := flist[i+1];
	if ffrun * fdim(next) > N then
            if N mod ffrun <> 0 or not CanTensorSplit(next, N/ffrun) then
		return false;
        #Error("can't merge incompatible tensor chains (could not split)",
         #     "N=", N, " ffrun=", ffrun, " next=", next);
            else 
		split := TensorSplit(next, N/ffrun);
		flist := Concat(flist{[1..i]}, split, flist{[i+2..len]});
		ffrun := ffrun * N/ffrun; # = N
            fi;
	else 
            ffrun := ffrun * fdim(next);
	fi;
	i := i+1;
    od;
    #Print([i, flist], "\n");
    return [i, flist];
end;

full_merge_tensor_chains := function(target, ff, gg, compose, ftensor, gtensor, fid, gid, fdom, gran) 
   local i, j, nf, ng, res, ffrun, ggrun, ibegin, jbegin, split;
   nf := Length(ff);    ng := Length(gg);
   res := []; i:=1; j:=1;

   while i <= nf or j <= ng do
       # handle domain=1 or range=1, which are always mergeable
       if (i<=nf and fdom(ff[i])=1) or (j<=ng and gran(gg[j])=1) then
	   if i<=nf and fdom(ff[i]) = 1 then
               Add(res, fid(ff[i]));
               i := i + 1; 
	   fi;
	   if j<=ng and gran(gg[j]) = 1 then
               Add(res, gid(gg[j]));
               j := j + 1;
	   fi;

       # try to combine terms to get a match
       elif (i <= nf and j <= ng) then
	   if AnySyms(fdom(ff[i]), gran(gg[j])) then
               return false;
           elif fdom(ff[i]) = gran(gg[j]) then
               Add(res, compose(ff[i], gg[j]));
	   else 
               ibegin := i;
               jbegin := j;
               if fdom(ff[i]) < gran(gg[j]) then
		   if CanTensorSplit(gg[j], fdom(ff[i])) then
		       split := TensorSplit(gg[j], fdom(ff[i]));
		       gg := Concat(gg{[1..j-1]}, split, gg{[j+1..ng]});
		       ng := ng + 1;
		   else 
		       split := try_split(i, ff, fdom, gran(gg[j]));
		       if split=false then return false; fi;
		       i := split[1];
		       ff := split[2]; nf := Length(ff);
		   fi;
               else
		   if CanTensorSplit(ff[i], gran(gg[j])) then
		       split := TensorSplit(ff[i], gran(gg[j]));
		       if split=false then return false; fi;
		       ff := Concat(ff{[1..i-1]}, split, ff{[i+1..nf]});
		       nf := nf + 1;
		   else 
		       split := try_split(j, gg, gran, fdom(ff[i]));
		       if split=false then return false; fi;
		       j := split[1];
		       gg := split[2]; ng := Length(gg);
		   fi;
               fi;
               Add(res, compose(ftensor(ff{[ibegin..i]}), gtensor(gg{[jbegin..j]})));
	   fi;
	   i := i+1;
	   j := j+1;
       else 
	   return false;
       fi;
   od;
   target.val := res;
   return res;
end;


# *******************************************************************
# make sure ff[i] is not a diag function, where merging chains doesn't make sense (does it?)
fully_compat_tensor_chains := (ff,gg,fdom,gran) -> 
    Length(ff) = Length(gg) and
    ForAll([1..Length(ff)], i -> ff[i].range()<>false and
                             fdom(ff[i])=gran(gg[i]));

merge_fc_tensor_chains := (ff,gg,combine) -> 
   List([1..Length(ff)], i -> combine(ff[i], gg[i]));

# this assumes compat_domain_range compatibility
compat_tensor_chains := (f,g,fdom,gran) -> let(
    ff := Filtered(f, c->let(d:=fdom(c), IsSymbolic(d) or d > 1)),
    gg := Filtered(g, c->let(r:=gran(c), IsSymbolic(r) or r > 1)), 
    fully_compat_tensor_chains(ff, gg, fdom, gran));

# this assumes compat_domain_range compatibility
merge_tensor_chains := function(ff, gg, compose, fidentity, gidentity, fdom, gran) 
   local i, j, nf, ng, res;
   nf := Length(ff);    ng := Length(gg);
   res := []; i:=1; j:=1;

   while i <= nf or j <= ng do
       if (i <= nf and j <= ng) and fdom(ff[i]) = gran(gg[j]) then
       Add(res, compose(ff[i], gg[j]));
       i := i+1;
       j := j+1;
       else
       if i<=nf and fdom(ff[i]) = 1 then
           Add(res, fidentity(ff[i]));
           i := i + 1;
       elif j<=ng and gran(gg[j]) = 1 then
           Add(res, gidentity(gg[j]));
           j := j + 1;
       else Error("can't merge incompatible tensor chains");
       fi;
       fi;
   od;
   return res;
end;
