
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Latex := x -> Cond(IsRec(x) and IsBound(x.latex), x.latex(), 
                   IsRec(x) and IsBound(x.cprint), x.cprint(), 
                   Print(x));
Cprint := x -> When(IsRec(x) and IsBound(x.cprint), x.cprint(), Print(x));

_infix_print := function ( lst, sep, prfunc )
    local  first, c;
    first := true;
    for c  in lst  do
        if first  then  first := false;
        else            Print(sep);  fi;
        prfunc(c); 
    od;
    return "";
end;

Gath.latex := self >> Print("\\Gath{", self.func.latex(), "}");
Scat.latex := self >> Print("\\Scat{", self.func.latex(), "}");
Diag.latex := self >> Print("\\Diag{", self.element.latex(), "}");
Prm.latex := self >> Print("\\Prm{", self.func.latex(), "}");

FuncClass.latex := self >> Print("\\f", self.name, 
    DoForAll(self.params, p -> Print("{",Cprint(p),"}")));

FDataOfs.latex := self >> Print("\\f", self.name, 
    DoForAll(self.rChildren(), p -> Print("{",Cprint(p),"}")));

HH.latex := self >> PrintEvalF("\\f$1{$2}{$3}{$4}{$5}", 
    self.name, ()->Latex(self.params[1]), ()->Latex(self.params[2]), ()->Latex(self.params[3]), 
    ()->_infix_print(self.params[4],", ",Latex));
BHH.latex := self >> PrintEvalF("\\f$1{$2}{$3}{$4}{$5}", 
    self.name, ()->Cprint(self.params[1]), ()->Cprint(self.params[2]), ()->Cprint(self.params[3]), 
    ()->_infix_print(self.params[4],", ",Cprint));
HHZ.latex := self >> PrintEvalF("\\f$1{$2}{$3}{$4}{$5}", 
    self.name, ()->Cprint(self.params[1]), ()->Cprint(self.params[2]), ()->Cprint(self.params[3]), ()->_infix_print(self.params[4],", ", Cprint));


NonTerminal.latex := self >> Print("\\n", self.name, 
    DoForAll(self.params, p -> Print("{",Latex(p),"}")));

Sym.latex := FuncClass.latex;

ISum.latex := self >> PrintEvalF("\\ISum{$1 0}{$2}{$3}", 
   () -> When(IsBound(self.var), Print(self.var, "=")), 
   () -> Latex(self.domain-1),
   () -> self.child(1).latex());

Compose.latex := self >> DoForAll(self.children(), c->c.latex());

BaseContainer.latex := self >> PrintEvalF("\\$1{$2}", self.name, ()->self.child(1).latex());

BaseOperation.latex := self >> Print("\\", self.name, DoForAll(self.children(), c->Print("{",c.latex(),"}")));

SUM.latex := self >> Chain(
    DoForAllButLast(self.children(), c->Print(c.latex(), "+")),
    Last(self.children()).latex());

gammaTensor.latex := self >> InfixPrint(self.children(), "\\boxtimes", c->c.latex());
fTensor.latex := self >> InfixPrint(self.children(), "\\otimes", c->c.latex());
fDirsum.latex := self >> InfixPrint(self.children(), "\\oplus", c->c.latex());
fCompose.latex := self >> InfixPrint(self.children(), "\\circ", c->c.latex());

Blk.latex := self >> self.print(0,0);
