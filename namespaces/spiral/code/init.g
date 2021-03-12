
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

# Intermediate Representation of Code 
# -----------------------------------
# This package defines data structures for intermediate
# representation of code expressions, statements, and 
# data types. 
#
# Main classes: 
#   Value - known fixed value, for example V(1.2)
#   Typ - data type, for example TDouble
#   Loc - memory location, e.g. a variable or an indexed array expression
#   Exp - expression, e.g. add(V(1), x)
#   Command - a statenent that performs an action, e.g. assign(a, add(V(1), b))
#   Lambda - symbolic representation of a function (can be used in Sigma-SPL)
#
# Exp ::
#    .args = arguments
#
# Loc subclasses:
#    var(<id>)         - variable 
#      .id       = string
#      .fresh()  = create fresh "xN" variable
#      .fresh_id(<prefix>) = create fresh "prefixN" variable
#
#    nth(<loc>, <idx>) - index <idx> (0-based) of an array location <loc>
#      .loc  
#      .idx  
#
# Exp subclasses:
#    add(<Exp>, <Exp>)
#    sub(<Exp>, <Exp>)
#    mul(<Exp>, <Exp>)
#
# Typ subclasses: TDouble, TInt, TComplex
#
# Command subclasses:
#    assign(<loc>, <exp>)
#       .loc = location
#       .exp = expression
#    skip
#    chain(<cmd1>, <cmd2>, ...)  OR  chain(<cmdlist>)
#       .cmds
#    decl
#       .vars
#       .cmd
#    data(<var>, <value>, <cmd>)
#       .var
#       .value
#       .cmd
#    loop(<var>, <range>, <cmd>)
#       .var
#       .range
#       .cmd
#    IF(<cond>, <then_cmd>, <else_cmd>)
#       .cond
#       .then_cmd
#       .else_cmd
#
# Examples:
#    Import(code);
#    cmd1 := assign(var.fresh(), add(var("x1"), sub(2, var("x3"))));
#
#@P

Import(rewrite);

Declare(EvalScalar, IsScalar, Scalars, IsSymbolic, IsLambda, IsFunction, 
        IsFuncExp, IsPosInt0Sym, IsArrayT, IsRealT, IsComplexT, IsVecT, TString, var,
        IsCommand, IsIntT, TBool);
Declare(_rank, _downRankFull, _convType, sizeof, _unwrap, _stripval, data);

Include(free);
Include(conhash);
Include(types);
Include(unify);
Include(ir);
Include(param);
Include(constraints);
Include(lambda);
Include(scalar); # old file, needed for compatibility
Include(lattice);
Include(propagate_types);
Include(symdiv);
Include(hof);
Include(gen);
Include(hacks);
Include(sreduce);

RulesStrengthReduce.__avoid__ := [Value];
RulesExpensiveStrengthReduce.__avoid__ := [Value];

Loc.visitAs     := "Loc";
Exp.visitAs     := "Exp";
Command.visitAs := "Command";

# Reset global counters, etc., used by code generator
# and release associated memory
ResetCodegen := function()
	var.flush();
	FlushConsts();
end;


