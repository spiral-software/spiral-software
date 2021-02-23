
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# misc functions for hamlet stuff


# takes the permutation in SPL, returns the bit matrix
# extends some of the Peter's stuff, -BA
SPLtoBitMatrix := function(perm)
	local m,bit;
	
	m := MatSPL(perm);

	bit := BitMatrixToInts(PermMatrixToBits(m));
	
	return bit;
	
end;


SPLtoLocalMemConf := function(perm,w)
	local opts,path,t,filename,cmdString;
	
	opts := InitStreamUnrollHw();
	
	path := Concat("/tmp/spiral/", String(GetPid()), "/");
	filename := "perm";
    MakeDir(path);
	
	BRAMPerm.print := (self,i,is) >> Print(self.name, "Mem(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")");
	
	t := streamGen(perm.withTags([AStream(w)]),opts);
	
	PrintTo(ConcatenationString(path, filename, ".spl"), HDLPrint(1024, t.dims(), -1, t));
	
    if (_splhdl_path = "") then
		Error("Error: path to SPLHDL compiler not set: paradigms.stream._splhdl_path is not bound.");
    fi;

    cmdString := ConcatenationString(_splhdl_path, " ", path, filename, ".spl");
	
	Exec(cmdString);
	
	BRAMPerm.print := (self,i,is) >> Print(self.name, "(", BitMatrixToInts(self._children[1]), ", ", BitMatrixToInts(self._children[2]), ", ", self.streamSize, ")");
	
	return path;
end;

