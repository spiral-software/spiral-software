# -*- Mode: shell-script -*- 

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

GlobalPackage(spiral.util);

SaveTable := rec();

#F save(<var>)
#F    Saves the value of the variable <var> in the SaveTable.
#F    The value can be retrieved later with 'restore(<var>)'. 
#F    save/restore pairs can be nested inside each other.
#F
save := UnevalArgs(
    function(var)
       Constraint(Type(var)=T_VAR);
       if not IsBound(SaveTable.(NameOf(var))) then
	   SaveTable.(NameOf(var)) := [ Eval(var) ];
       else
	   Add(SaveTable.(NameOf(var)), Eval(var));
       fi;
    end
);

#F restore(<var>)
#F    Restores the value of the variable <var> from the SaveTable.
#F    The value must have been saved earlier with 'save(<var>)'. 
#F    save/restore pairs can be nested inside each other.
#F
restore := UnevalArgs(
    function(var)
       local res;
       Constraint(Type(var)=T_VAR);
       if not IsBound(SaveTable.(NameOf(var))) then
	   Error("No value for <", NameOf(var), "> found in SaveTable");
       elif SaveTable.(NameOf(var)) = [] then
	   Error("No value for <", NameOf(var), "> found in SaveTable");
       else
	   res := Last(SaveTable.(NameOf(var)));
	   RemoveLast(SaveTable.(NameOf(var)), 1);
	   Assign(var, res);
	   return res;
       fi;
    end
);

store := UnevalArgs(
    function(var)
       local nam, file;
       Constraint(Type(var)=T_VAR);
       nam := NameOf(var);
       file := ConcatenationString(nam, ".g");
       PrintTo(file, "ImportAll(spiral);\n");
       AppendTo(file, nam, " := ", Eval(var), ";");
    end
);
