
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#F CondPat(<obj>, <pat1>, <result1>, ..., <patN>, <resultN>, [<resultElse>])
#F
#F CondPat is a conditional control-flow construct similar to Cond, but instead
#F of normal boolean conditions it uses patterns that are matched against its
#F first argument <obj>. CondPat returns the result that corresponds to the
#F first matched pattern. If nothing matches <resultElse> is returned.
#F
#F Note: resultElse is optional
#F 
#F CondPat is implemented as a function with delayed evaluation, and will
#F only evaluate the result that corresponds to the matched patterns, all
#F other results will not be evaluated, for example:
#F
#F CondPat(mul(X,2), [mul, @, 2], Print("yes"), [add, @, 2], Print("no"));
#F
#F will only execute Print("yes") -- as expected.
#F

CondPat := UnevalArgs(function(arg)
    local usage, i, pat, obj;
    
	usage := "CondPat(<obj>, <pat1>, <result1>, ..., <patN>, <resultN>, <resultElse>)";
    
	if Length(arg) < 3 then 
		Error(usage); 
	fi;

    obj := Eval(arg[1]);
    i := 2;
    while i <= Length(arg)-1 do
        pat := Eval(arg[i]);
		if PatternMatch(obj, pat, empty_cx()) then 
			return Eval(arg[i+1]); 
		fi;
		i := i+2;
    od;
    if i=Length(arg) then
		return Eval(Last(arg));
    else 
		Error("CondPat: No 'else' result, and no patterns match");
    fi;
end);
