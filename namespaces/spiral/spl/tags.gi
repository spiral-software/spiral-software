
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


IsTag := x -> IsRec(x) and IsBound(x.isTag) and x.isTag;

Declare(ANoTag);

####################################################################
#F AGenericTag - base class for tags, uses RewritableObject features
#F    which provides
#F      __call__
#F      print
#F      rSetChild
#F      rChildren
#F      lessThan
#F      equal
#F
#F Constructor saves all parameters into .params.
#F  Please read Doc(RewritableObject).
#F
Class(AGenericTag, RewritableObject, rec(
    isTag := true,
    kind := self >> ObjId(self),

    # YSV: this should be called when we transpose the parent structure
    transpose := self >> self,   

    isSticky  := false,
    right     := self >> When(IsBound(self.isRight), self.isRight, false),
    left      := self >> When(IsBound(self.isRight), not self.isRight, false),
    bisided   := self >> not IsBound(self.isRight),

    leftChild  := self >> When(IsBound(self.isLeftChild),  self.isLeftChild,  false),
    rightChild := self >> When(IsBound(self.isRightChild), self.isRightChild, false),

    edge       := self >> When(IsBound(self.isEdge),  self.isEdge,  false),

    leftEdge   := self >> self.edge() and self.leftChild(),
    rightEdge  := self >> self.edge() and self.rightChild(),

    # distCompose(<chlist>, <opts>)
    #   Distributes tag over Compose(<chlist>).
    #   Returns list of tags to assign to each composition member.
    #   ANoTag is thrown out later during normalization.
    distCompose := (self, chlist, opts) >>
        Cond( self.left(),  
                  [self] :: Replicate(Length(chlist)-1, ANoTag),
              self.right(), 
                  Replicate(Length(chlist)-1, ANoTag) :: [self],
              # else bisided
                  Replicate(Length(chlist), self) ),

));

Class(ASingletonTag, AGenericTag, rec(
    params := [],
    from_rChildren := (self, rch) >> self,
    kind := self >> self
));

#F ANoRecurse() -- tag that prevents further recursion. 
#F
#F Currently only used in autolib.
#F
Class(ANoRecurse, AGenericTag);

#F ANoTag - denotes absence of tags
#F
Class(ANoTag, ASingletonTag, rec());

#F ATopLevel - used to denote that special top level processing should be used,
#F             generally param[1] denotes at which 'level' the tag should be
#F             dropped.
#F
#F             Marek uses it in his WHT/bit-perms expansions. This tag will
#F             be moved elsewhere soon.
Class(ATopLevel, AGenericTag, rec());


####################################################################
#F Mixin class for taggable objects
#F
#F This object/class will add tagging functionality to whatever object inherits it.
#F Like all mixin style classes, it is not meant to be the sole parent class.
#F
Class(TaggedObjectMixin, rec(
    # this is the actual list of tags. This array should not be accessed directy, rather
    # the methods should be used to parse it.
    tags := [],

    # returns the first tag or ANoTag
    firstTag := self >> When(Length(self.tags) >= 1, self.tags[1], ANoTag),

    # returns true if first tag is has 'tag' object id
    firstTagIs := (self, tag) >> self.firstTag().kind()=tag,

    # returns the tags
    getTags := self >> self.tags,

    # returns true if the object has tags
    hasTags := self >> self.tags <> [],

    # returns a copy of the object with 'tags' added on to any existing tags.
    withTags := (self, tags) >> Checked(IsList(tags), self.setTags(self.tags :: tags)),

    # returns a copy of the object with tags given by 'tags'
    setTags := (self, tags) >> Checked(IsList(tags), CopyFields(self, rec(tags := tags))),

    # return a copy of the whole object without the first tag
    withoutFirstTag := self >> self.setTags(Drop(self.tags, 1)),

    # return a copy of the whole object without the tags given by object id 'tag'
    withoutTag := (self, tag) >> Checked(IsTag(tag), self.setTags(Filtered(self.tags, e -> e.kind() <> tag))),

    # returns true when the object id of at least one of the tags is 'tag'
    hasTag := (self, tag) >> Checked(IsTag(tag), ForAny(self.tags, e -> e.kind() = tag)),

    hasAnyTag := (self, tags) >> Checked(IsList(tags), ForAny(self.tags, e -> e.kind() in tags)),

    # returns true when tag number 'n' has the object id given by 't'
    isTag := (self, n, t) >> When(Length(self.tags) >= n, self.tags[n].kind() = t, false),

    numTags := (self) >> Length(self.tags),

    dropTags := (self) >> CopyFields(self, rec(tags := [])),

    # 
    #F getTag
    #
    # get the tag by name or number, function has 3 forms:
    #   getTag(2)           returns 2nd tag
    #   getTag(ASomeTag)    if there is only 1 ASomeTag, returns it, if >1, returns array
    #   getTag(ASomeTag, 2) returns 2nd ASomeTag
    # 
    # in case of error, getTag returns false
    #
    getTag := meth(arg)
        local s, t, n;

        Constraint(Length(arg) = 2 or Length(arg) = 3);

        s := arg[1];

        # arg[2] is a number.
        if IsInt(arg[2]) then
            return When(Length(s.tags) >= arg[2],
                s.tags[arg[2]],
                false);

        # arg[2] must be a tag id.
        else
            t := Filtered(arg[1].tags, e -> e.kind() = arg[2]); 

            # if tag and number are specified
            if Length(arg) = 3 then
                return When(Length(t) >= arg[3],
                    t[arg[3]],
                    false
                );
            
            else
                return When(t = [], 
                    false,
                    When(Length(t) = 1,
                        t[1],
                        t
                    )
                );
            fi;
        fi;
    end,

    getAnyTag := (self, tags) >> First(self.tags, x->x.kind() in tags)
        
));

####################################################################
#F Tagged non-terminal. This should be used for all new taggable
#F nonterminals.
#F
#F This supercedes the old system which puts tags into .params field
#F
#F Tags on nonterminals contain information that is beyond the mathematical
#F details of the object. Often tags contain information about hardware
#F which is used to direct the breakdown.
#F
Class(TaggedNonTerminal, TaggedObjectMixin, NonTerminal, rec(
    _short_print := true,
    #--------- Transformation rules support --------------------------------
    from_rChildren := (self, rch) >> let(
        len := Length(rch),
        transposed := rch[len-1],
        tags := rch[len],
        t := ApplyFunc(ObjId(self), rch{[1..len-2]}),
        tt := When(transposed, t.transpose(), t),
        attrTakeA(tt.withTags(tags), self)
    ),

    rChildren := self >>
        Concatenation(self.params, [self.transposed, self.tags]),

    rSetChild := meth(self, n, newChild)
        local l;
        l := Length(self.params);
        if n <= l then
            self.params[n] := newChild;
        elif n = l+1 then
            self.transposed := newChild;
        elif n = l+2 then
            self.tags := newChild;
        else Error("<n> must be in [1..", l+2, "]");
        fi;
        # self.canonizeParams(); ??
        self.dimensions := self.dims();
    end,

    # this print method OVERRIDES the print method in NonTerminal.
    # here, we print the TAG INFO.
    print := (self, i, is) >> Print(
	Inherited(i, is),
        When(IsBound(self.tags) and IsList(self.tags) and self.tags<>[], 
             Print(".withTags(", self.tags, ")"))), 
    
    takeTA := (self, o) >> self.setTags(o.getTags()).takeAobj(o),
));
