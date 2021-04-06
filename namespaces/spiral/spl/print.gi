
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F Printing of SPLs
#F ----------------
#F

#F SPLOps.Print( <spl> [, <indent> , <indentStep> ] )
#F   prints <spl> with <indent>. Further indenting is done
#F   in steps of size <indentStep>. The default is
#F   indent = 0, indentStep = 2.
#F
SPLOps.Print := function ( arg )
    local S, indent, indentStep;
    if Length(arg) = 1 then   S := arg[1]; indent := 0; indentStep := 2;
    elif Length(arg) = 3 then S := arg[1]; indent := arg[2]; indentStep := arg[3];
    else Error("usage: SPLOps.Print( <spl> [, <indent> , <indentStep> ] )");
    fi;
    Constraint(IsInt(indent) and indent >= 0);
    Constraint(IsInt(indentStep) and indentStep >= 0);
    if IsInt(S) then
	Print(S);
    else
	S.print(indent, indentStep);
    fi;
end;

_CompactPrintSPL := rec(
    doPrintCutoff := true,

    indentWithLines := function(indent, indentStep)
        local x, beg;
	if indent < indentStep - 2 then
	    Print(Blanks(indent));
	else
	    beg := indent mod indentStep;
	    Print(Blanks(beg));
	    x := beg+1;
	    while x < indent - indentStep do
	        Print(" |", Blanks(indentStep-2));
		x := x + indentStep;
	    od;
	    Print(" +"); x := x + 2;
	    while x < indent do
	        Print("-");
		x := x + 1;
	    od;
	fi;
    end,

    indentBlank := function(indent, indentStep) 
        Print(Blanks(indent)); 
    end,

    print := meth(self, spl, maxDepth, indentFunc, indent, indentStep)
        local c;
	if maxDepth > 0 then 
	    indentFunc(indent, indentStep);
	    if IsBound(spl.symbol) then Print(spl.symbol, " ", spl.params, "\n"); 
	    else Print(spl.name, "\n"); 
	    fi;
	    if IsBound(spl.children) then
		for c in spl.children() do
	            self.print(c, maxDepth-1, indentFunc, indent+indentStep, indentStep);
		od;
	    fi;
	elif self.doPrintCutoff then
	    indentFunc(indent, indentStep);
	    Print("...\n");
	fi;
    end
);

CompactPrintSPL := function(spl, maxDepth)
    Constraint(IsSPL(spl));
    Constraint(IsInt(maxDepth) and maxDepth >= 0);
    _CompactPrintSPL.print(spl,maxDepth, _CompactPrintSPL.indentBlank, 0,5);
end;

CompactPrintSPLTree := function(spl, maxDepth)
    Constraint(IsSPL(spl));
    Constraint(IsInt(maxDepth) and maxDepth >= 0);
    _CompactPrintSPL.print(spl,maxDepth, _CompactPrintSPL.indentWithLines, 0,5);
end;

CompactPrintSPLNode := function(spl)
    Constraint(IsSPL(spl));
    if IsBound(spl.symbol) then Print(spl.symbol, " ", spl.params); 
    else Print(spl.name); 
    fi;
end;
