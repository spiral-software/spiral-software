
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


# ----------------------------------------------------------------------------------
# Visitor - base class for objects implementing the visitor pattern from Gamma book
# ----------------------------------------------------------------------------------
# This class basically implements a somewhat modified variant of the original
# Visitor pattern from the book by Gamma, et al. "Design Patterns".
#
# Usage example:
#
# define the following class
#
# Class(LispGen, Visitor, rec(
#     add := (self, o) >> Print("(+ ", self(o.args[1]), " ", self(o.args[2]), ")"),
#     mul := (self, o) >> Print("(* ", self(o.args[1]), " ", self(o.args[2]), ")"),
#     sub := (self, o) >> Print("(- ", self(o.args[1]), " ", self(o.args[2]), ")"),
#     var := (self, o) >> Print("(var ", o.id, ")"),
#     Value := (self, o) >> Print("(value ", o.v, ")")
# ));
#
# spiral> LispGen(4*X+2);
# (+ (* (value 4) (var X)) (value 2))spiral> 
#
# Note that instead of using 'self' as a function (which invokes __call__),
# we can use self.visit, and make __call__ a constructor for LispGen.
#
# Ie., the methods would look like
#     add := (self, o) >> Print("(+ ", self.visit(o.args[1]), " ", self.visit(o.args[2]), ")"),
#
Class(Visitor, rec(
   __call__ := arg >> ApplyFunc(arg[1].visit, arg{[2..Length(arg)]}),

    # this is a rewrite of the lambda expression. The lambda was getting
    # way too long. This implements the double dispach for the visitor
    # class. 
    visit := meth(arg)
        local self, o, len;

        Constraint(Length(arg) >= 2);

        self := arg[1];
        o := arg[2];
        len := Length(arg);

        if IsRec(o) or IsList(o) then
            o := ObjId(o);
            if IsBound(self.(o.name)) then
                ApplyFunc(self.(o.name), arg{[2..len]});

            # this is just here to trap the hacks, rather than silently fail
            elif IsBound(o.visitAs) then
                Error("visitAs has been removed. Don't use it.");

            else
		        Error("Cannot visit <arg[2]>. Visitor ", self, 
                " does not have field '", o.name, "'", 
                    When(IsBound(o.visitAs), 
			            Concat(" or ",o.visitAs, " (from .visitAs)"), 
                        ""
                    )
                );
            fi;
        else
            ApplyFunc(self.atomic, arg{[2..len]});
        fi;
    end
));

#F
#F# getVisitAs
#F
#F get the .visitAs if it exists.
#F

getVisitAs := function(v, o)


    o := ObjId(o);

    # if this object has a visitor, or if visitAs is unbound
    if IsBound(v.(o.__name__)) or not IsBound(o.visitAs) then
        return false;
    fi;

    return o.visitAs;
end;

#F HierarchicalVisitor -- same as Visitor, but does not use .visitAs, 
#F   instead traverses the super class chain 
#F
Class(HierarchicalVisitor, rec(
   __call__ := arg >> ApplyFunc(arg[1].visit, arg{[2..Length(arg)]}),

#F the hierarchical visitor function. 
#F 
#F we consider the objects by level, in order of the objects in __bases__
#F
#F here's an example:
#F objA
#F  |---parA
#F  | |---parA1
#F  | \---parA2
#F  \---parB
#F
#F traverse order is: objA parA parB parA1 parA2
#F
    warnings := Set([]),
    paranoiaMode := false, # <-- if this is true, then .visit() aborts when there is a mismatch of 
                          # the method it finds with what .visitAs prescribes, otherwise,
                          # it silently adds the mismatch to .warnings and continues

    visit := meth(arg)
        local self, o, orig, len, parents, newparents, v;
      
        Constraint(Length(arg) >= 2);

        self := arg[1];
        o   := arg[2];
        len := Length(arg);

        # lists are handled with .ListClass, records with .<objid>, 
        # other objects with .atomic
        if not (IsRec(o) or IsList(o)) or IsString(o) then
            return ApplyFunc(self.atomic, arg{[2..len]});

        # objects are handled here
        else
            # this will be eventually removed, but for the time being,
            # we try to figure out the object which would be selected
            # by .visitAs and compare it to the object which is chosen
            # hierarchically. If there is a difference, we throw an
            # error.

            v := getVisitAs(self, o);

            # paranoia check to make sure we have an object
            Constraint(IsList(o) or (IsRec(o) and IsBound(o.__name__)));

            parents := Cond(IsList(o), [ObjId(o)], ShallowCopy(o.__bases__));

            # traverse the parent tree as given in the function comments
            for o in parents do
                if IsBound(self.(o.__name__)) then

                    # here is our paranoia check which will eventually
                    # be removed after the transition is complete.
                    if (v <> false and v <> o.__name__) then
                        if self.paranoiaMode then
                            Error("visitAs object is different from object picked by hierarchy traversal. Call Marek.");
                        else
                            AddSet(self.warnings, [self,v,o]); 
                        fi;
                    fi;

                    return ApplyFunc(self.(o.__name__), arg{[2..len]});
                elif IsBound(o.__bases__) then
                    Append(parents, o.__bases__);
                fi;
            od;

            return Error("Cannot visit <arg[2]>. visitAs was ", v);
        fi;
    end,

    #F
    #F getBases()
    #F
    #F build a list of all the bases according to the order
    #F given in the .visitAs method
    #F
    getBases := meth(self)
        local b, i;

        b := [self];
        i := 1;

        while i <= Length(b) do
            Append(b, ShallowCopy(b[i].__bases__));
            i := i + 1;
        od;

        return b;
    end,

    #F
    #F showMatches(obj)
    #F
    #F returns the ordered list of visitors triggered by this object.
    showMatches := meth(self, o)
        local b, bb, res, i, m;

        Constraint(IsRec(o));
        
        b := self.getBases();

        bb := ShallowCopy(o.__bases__);
        res := [];
        i := 1;

        while i <= Length(bb) do
            m := Filtered(b, e -> bb[i].name in UserRecFields(e));

            if m <> [] then
                Append(res, [bb[i]] :: m);
            fi;

            Append(bb, ShallowCopy(bb[i].__bases__));
            i := i + 1;
        od;

        return res;
    end,
));


#F HierarchicalVisitorCx -- HierarchicalVisitor with context accessible through
#F   self.cx field. Drawback - start visitor using <walk> method instead of 
#F   __call__.
#F
#F   Ex: MyVisitor.walk(tree, arg1, arg2);
#F

Class(HierarchicalVisitorCx, HierarchicalVisitor, rec(

    walk := (arg) >> ApplyFunc(CopyFields(arg[1], rec( 
        cx := empty_cx(), _current := false )), Drop(arg, 1)),

    visit := meth(arg)
        local self, parent, res;

        self          := arg[1];
        parent        := self._current;
        self._current := arg[2];

        if parent=false then
            res := ApplyFunc(Inherited, Drop(arg, 1));
        else
            cx_enter(self.cx, parent);
            res := ApplyFunc(Inherited, Drop(arg, 1));
            cx_leave(self.cx, parent);
        fi;
        
        self._current := parent;

        return res;
    end,
));
