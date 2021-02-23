
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


processCRP := meth(partA, partB)
    local size, switches, next, modA, modB, loc, setA, choice;
    size := Length(partA);
    
    modA := List([1..size], i-> Mod(partA[i], size));
    modB := List([1..size], i-> Mod(partB[i], size));
    
    setA := [];

    switches := List([1..size], i->-1);
    next := -1;
        
    while (-1 in switches) do
    
        # if next = -1, then we can make an arbitrary decision.
        if (next < 0) then
            # find the next switch that isn't set.
            choice := PositionProperty(modA, i-> i <> -1);
            
            switches[choice] := 0;
            Append(setA, [modA[choice]]);

            # We just assigned modA[choice] to partition A.
            # Thus, we precluded modB[choice] from being in parition A.
            # So, we need to check.  If modB[choice] is not already in 
            # partition A, then we need to make it our next choice.
            
            if (modB[choice] in setA) then
                next := -1;
            else
                next := modB[choice];
            fi;            
            modA[choice] := -1; modB[choice] := -1;
            
        else
            # if next >= 0, then it represents the 'mod' number we must
            # assign to partition A.
            
            # the next value must be in the set of unchosen values in one
            # of the two partitions.
            if (next in modA) then
                choice := Position(modA, next);
                switches[choice] := 0;
                Append(setA, [modA[choice]]);

                if (modB[choice] in setA) then
                    next := -1;
                else
                    next := modB[choice];
                fi;
                
            else
                choice := Position(modB, next);
                switches[choice] := 1;
                Append(setA, [modB[choice]]);
                
                if (modA[choice] in setA) then
                    next := -1;
                else
                    next := modA[choice];
                fi;
            fi;
            modA[choice] := -1; modB[choice] := -1;
        fi;
 
    od;

    return switches;
end;

# Given a permutation permIn, determine the switch settings for the
# un-simplified \Omega\Omega^{-1} network.  
#
# See: K. Y. Lee.  On the rearrangeability of 2(\log_2n)-1 stage
# permutation networks.  IEEE Transactions on Computers,
# C-34(5):412-425, May 1985.

# We do not simplify the network, because my network generation code
# will simplify all unneeded switches.

routePerm := meth(permIn)
    local n, ss, ns, p, p2, col, numP, partSize, partA, partB, i, j, k, partStride, tmpSw, swCol, tmp;
    
#    Print("Routing perm: ", permIn, "\n");

    n := Length(permIn);

    # The perm comes in as row values.  I.e., PermRowVals[i] -> i.
    # We need to transpose this.    
    p := List([1..n], i->0);
    for i in [0..n-1] do p[permIn[i+1]+1] := i; od;
    
    # This is suboptimal.  For now, I will simply let the unnecessary
    # switches be all-zero.
    ns := n/2 * (LogInt(n,2) + LogInt(n,2)-1);
    
    ss := [];
    
    for k in [0..LogInt(n,2)-2] do
        col := LogInt(n,2)-1-k;
    
        # Permute output values by L(n,n/2)
        p2 := List([1..n], i -> p[L(n,n/2).at(i-1).ev()+1]);

        # Now we need to set the switches for the E^{col}_{R} column.        
        numP := 2^k;
        partSize := n/(2*numP);
        partStride := 2*numP;
        
        swCol := List([1..n/2], i -> -1);
        tmpSw := [];
        for i in [0..numP-1] do
            partA := [];
            partB := [];
            for j in [1..partSize] do
                Append(partA, [p2[partStride*(j-1)+1+(2*i)]]);
                Append(partB, [p2[partStride*(j-1)+2+(2*i)]]);                
            od;

            Append(tmpSw, processCRP(partA, partB));
            
        od;
 
        swCol := List([1..n/2], i -> tmpSw[L(n/2,n/(2^(k+1))).at(i-1).ev()+1]);
        Append(ss, swCol);
       
        # Now we have to perform the permutation based on the
        # switches we set.
        for i in [0..n/2-1] do
            if (swCol[i+1] = 1) then
                p[i*2+1] := p2[i*2+2];
                p[i*2+2] := p2[i*2+1];
            else
                p[i*2+1] := p2[i*2+1];
                p[i*2+2] := p2[i*2+2];
            fi;
        od;
    od;

    # Permute output values by L(n,n/2)
    p2 := List([1..n], i -> p[L(n,n/2).at(i-1).ev()+1]);    

    for k in [0..LogInt(n,2)-1] do

        swCol := [];
        # go through the n/2 switches.  at each switch, if the top input's value
        # has a 1 in bit k.
        for i in [0..n/2-1] do
            if (imod(idiv(p2[2*i+1], 2^k), 2).ev() = 1) then
                Append(swCol, [1]);
            else
                Append(swCol, [0]);
            fi;
        od;

        Append(ss, swCol);

        # Now we have to perform the permutation based on the
        # switches we set.
        for i in [0..n/2-1] do
            if (swCol[i+1] = 1) then
                p[i*2+1] := p2[i*2+2];
                p[i*2+2] := p2[i*2+1];
            else
                p[i*2+1] := p2[i*2+1];
                p[i*2+2] := p2[i*2+2];
            fi;
        od;

        # Permute output values by L(n,2)
        p2 := List([1..n], i -> p[L(n,2).at(i-1).ev()+1]);    

    od;

    p := List([0..n-1], i->i);
    if (p <> p2) then
        Error("Failed at configuring switching network.");
    fi;

#    Print("Configuration: ", ss, "\n");

    return ss;
end;



# Given a permutation permIn, determine the switch settings for the
# \Omega^{-1} network.  
#
# We will return -1 if the permutation cannot be mapped to that network.
# In this case, then, you should use routePerm to map it to the
# \Omega\Omega^{-1} network.

# For now, we will cheat and use the \Omega\Omega^{-1} routing algorithm.
# It appears that when we use this algorithm on a permutation that can be
# permformed on the \Omega^{-1}, then the configuration bits are simply
# all zero for the first \Omega portion of the network.  However, I haven't 
# verified this and I haven't looked at the routePerm code in many years.

# I have not proven this will be the case, but it seems to work on the
# circular shift permutations I am currently using.

# All circular shifts will be Omega^{-1} passable by Theorems 3  and 13 of "Access
# and Alignment of Data in an Array Processor," Duncan H. Lawrie, IEEE Tr.
# on Computers, C-24 (12), 1975.
#
# Theorem 3: Duncan says that if you can represent your mapping as:
# P_N(a, b, c, d, e) = {(ax+b, cx+d) | 0 \leq x < e}
# where ax+b represents the source (mod N) and cx+d represents the
# destination (mod N),
# then your mapping is \Omega passable if gcd(a,N) \leq gcd(c,N) and
# e \leq N/(gcd(c,N).
# For a cyclic shift, a and c will always = 1, b=0, d equals the number
# of positions to shift, and e=N.
# Since a and c are 1, gcd(a,N) and gcd(c,N) are both 1 and hence equal.
# Further, N/gcd(c,N) = N, and therefore e = N/gcd(c,N).

# Lastly we note that Theorem 13 says that if a mapping is \Omega passable
# it must be \Omega^{-1} passable.

routePermOmegaInv := meth(permIn)
    local full, n, ns, numSwitchesZero;

    full := routePerm(permIn);
    n := Length(permIn);
    ns := n/2 * (LogInt(n,2) + LogInt(n,2)-1);
    
    # If this permutation is Omega^{-1} routable, then the first numSwitchesZero
    # must be configured to zero.
    numSwitchesZero := n/2 * (LogInt(n,2)-1);
    
    if (Sum(full{[1..numSwitchesZero]}) <> 0) then
	return -1;
    fi;
	
    return full{[numSwitchesZero+1 .. ns]};
end;

cyclicShiftList := meth(size, shift)
    local t, res, t2;
    t := [0..size-1];
    shift := Mod(shift, size);
    res := t{[shift+1..size]};
    if (shift = 1) then
	t2 := [0];
    else
	t2 := t{[1..shift]};
    fi;

    Append(res, t2);
    return res;
end;


Class(omegaInvNetwork, BaseMat, SumsBase, rec(

    abbrevs := [(n, p, it) -> [n, p, it]],

    new := (self, n, perms) >> SPL(WithBases(self, rec(
		    dimensions := [n, n],
		    _children := [n, perms]
	     ))),
    rChildren := self >> self._children,
    rSetChild := meth(self, n, what) self._children[n] := what; end,
    child     := (self, n) >> self._children[n],
    children  := self >> self._children,

    
    print      := (self,i,is) >> Print(self.name, "(", self.child(1), ", ", self.child(2), ")"),

    dims := self >> [self._children[1], self._children[1]],

));

overlapAddCyclicShifts := meth(num_points, step_size, num_steps)
    local res, i, it;
    
    res := [];
    for i in [0 .. num_steps-1] do
       Append(res, [[routePermOmegaInv(cyclicShiftList(num_points, step_size*i))]]);
    od;

    return omegaInvNetwork(num_points, res);

end;

# generage Verilog for Inverse Omega network for a family of cyclic shift permutation, 
# given by C_n^(s*l), where l = 0 .. k.
# In code below, n is "num_points", s is "step_size", k is "num_steps".
#
# "bits" gives the number of bits per word.
# "name" gives the output filename, which will be created in /tmp/spiral/[PID] where 
# [PID] is Spiral's process ID.  You can find the current PID in Spiral by typing
# getPid();
#
# If you want to change this diretory, it is easy to do
# so; just edit the file hacks.gi in spiral/paradigms/stream/hacks.gi
#
# To generate an 8 point network that will perform C_8^0, C_8^1, ..., C_8^7, and
# store it in file.v, run:
#    genOverlapAddShift(8, 1, 8, 16, "file");
genOverlapAddShift := function(num_points, step_size, num_steps, bits, name)
    local opts;
    opts := InitStreamUnrollHw();
    HDLGen(overlapAddCyclicShifts(num_points, step_size, num_steps), 1, 0, bits, 0, 0, name);
end;