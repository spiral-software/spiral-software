
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


#############################################################################
#F
#F  sched.g                    Bag of Schedulers                Ernest Chan
#F
##

#############################################################################
##
##  Constants
## 
PRINTDAG := false;
RED := 0;           # INPUT NODES
BLACK := 1;         # TEMP NODES
BLUE := 2;          # OUTPUT NODES
GREEN := 3;         # WEIRD! no pred and no succ


############################################################################
##
##  Building the DAG
##
##  Node structure
##    cmd:     the original statement
##    pred:    a list of predecessors 
##    succ:    a list of successors 
##    id:      list of command outputs
##

############################################################################
##
#F  AddNode( <dag>, <cmd> ) . . makes a node for a variable/input/output
## 
AddNode := function(dag, cmd)
  local a;
  a := rec(
    id := Set(cmd.op_out() :: cmd.op_inout()),     # either var or nth
    index := Length(dag),
    pred := [],
    succ := [],
    cmd := cmd,
  ); 

  Add(dag, a);

  return a;
end;

############################################################################
##  
#F  GetNode( <dag> , <id> ) . . . . . . if node with the specified <id> is 
#F      in <dag>, then it is returned,  otherwise returns false
##
GetNode := function(dag, var)
  local i;
 
  i := Length(dag);
  while i >= 1 do
    if var in dag[i].id then
      return dag[i];
    fi;
    i := i-1; 
  od;

  return false;
end;


############################################################################
##
#F  AddEdges( <dag> , <cmd> ) . . . parses out the variables from 
#F      <exp>, and addes an edge from each of the variables to <node> in <dag> 
##
AddEdges := function(dag, cmd) 
    local v, n, ops_in, node;
    
    ops_in := Set(cmd.op_in() :: cmd.op_inout());

    node := AddNode(dag, cmd);

    for v in ops_in do
        n := GetNode(dag, v);
        if n <> false then
            Add(n.succ, node);
            Add(node.pred, n);
        fi;
    od;
end;


##############################################################################
##
#F  BuildDag( <chain> ) . . . . builds the Dag by examining 
#F      each command statement in <chain>
##
BuildDag := function(chain_object) 
    local i, j, k, dag, newnode, old_id, newnode_name, f, objid, assignstmt, node;
    if (ObjId(chain_object) = chain) then
        dag := [];
        for i in [1..Length(chain_object.cmds)] do
            # for each assign statement, make appropriate changes to the Dag have to 
            #   1) add nodes if not in the graph
            #   2) add edges            
            AddEdges(dag, chain_object.cmds[i]);
        od;
        return dag;
    else
        Print("in function BuildDag: argument not a chain structure");
    fi;
end;


##############################################################################
##
#F  PrintDag ( <dag> , <bool> ) . . . . if true, prints the dag with its orders,
#F      otherwise prints the dag with the names of the nodes                                  
##
PrintDag := function(dag, order)   
  local count, _PrintDag, __PrintDag;

  Print("digraph DAG {\n");     
  
  count := 0;

  _PrintDag := function(dag, order) 
    local i,j,templist;

    if Length(dag)<1 then return; fi;

    if IsList(dag[1]) then
      for i in [1..Length(dag)] do
        Print("  subgraph cluster");
        Print(count);
        count := count + 1;
        Print(" {\n");
        _PrintDag(dag[i], order);
        Print("  }\n");
      od;
    else 
      for i in [1..Length(dag)] do  
        Print("  ");
        if order then
          Print(dag[i].order);
        else
          Print("\"", dag[i].id, "\"");
        fi;
      od;
      Print("\n");
    fi;
  end;

  _PrintDag(dag, order);


  __PrintDag := function(dag,order)
    local i,j;
   
    if Length(dag)<1 then return; fi;
 
    if IsList(dag[1]) then
      for i in [1..Length(dag)] do
        __PrintDag(dag[i], order);
      od;         
    else 
      for i in [1..Length(dag)] do
        for j in [1..Length(dag[i].succ)] do
          Print("  ");
          if order then
            Print(dag[i].order);
          else 
            Print("\"", dag[i].id, "\""); 
          fi;
          Print(" -> ");
          if order then
            Print(dag[i].succ[j].order);
          else 
            Print("\"", dag[i].succ[j].id, "\"");        
          fi;
          Print("\n");
        od;
      od;
    fi;
  end;

  __PrintDag(dag, order);

  Print("}\n");
end;



##############################################################################
##
#F  OutputDag( <dag> , <string> ) . . . . . . . outputs a postscript file for 
#F      the dag with filename <string>.ps
##
OutputDag := function(dag, file)
  if PRINTDAG then 
    PrintTo("temp", PrintDag(dag, false));
    SYS_EXEC(Concat("dot -Tps -o",file," temp"));   
    SYS_EXEC("rm temp");
  fi;
end;


##############################################################################
##
#F  InitColor ( <dag> )  . . . . . . . . initialize color of nodes in a dag
##
InitColor := function(dag)
  local i, j, pred_exists, succ_exists;

  for i in [1..Length(dag)] do
    pred_exists := Filtered(dag[i].pred, e->e in dag) <> [];
    succ_exists := Filtered(dag[i].succ, e->e in dag) <> [];

    if pred_exists then
      if succ_exists then
        dag[i].color := BLACK;  
      else 
        dag[i].color := BLUE;
      fi;
    else
      if succ_exists then
        dag[i].color := RED;  
      else 
        Error("Not a DAG");
      fi;
    fi;
  od;    
  
end;

###############################################################################
##
#F  ColorPhase ( <dag> , <color> ) . . advances one step in the coloring phase
##
ColorPhase := function(dag, color)
  local i, j, pos, all_has_color, colorlist;

  colorlist := [];

  if color=RED then
    for i in [1..Length(dag)] do
      if dag[i].color = BLACK then 
        all_has_color := true;
        for j in dag[i].pred do
          if j in dag then
            if j.color<>RED then
              all_has_color := false;
            fi;
          fi;
        od;
        if all_has_color then
          Add(colorlist, i);
        fi;
      fi;
    od;
  elif color=BLUE then
    for i in [1..Length(dag)] do
      if dag[i].color = BLACK then 
        all_has_color := true;
        for j in dag[i].succ do
          if j in dag then
            if j.color<>BLUE then
              all_has_color := false;
            fi;
          fi;
        od;
        if all_has_color then
          dag[i].color := BLUE;
        fi;
      fi;
    od;
  fi;

  for i in [1..Length(colorlist)] do
    dag[colorlist[i]].color := color;  
  od;

end;


###############################################################################
##
#F  SplitDag( <dag> ) . . . . . splits a dag into two (roughly equal) halves
##
SplitDag := function(dag)
  local getComponent, set, newset, temp, answer, i;

  getComponent := function(dag, set, node)
    local i, j, pred, succ, tempset;
   
    pred := Filtered(node.pred, e->e in dag);
    succ := Filtered(node.succ, e->e in dag);
 
    if not node in set then
      Add(set, node);
      node.taken := true;
    fi;
    for i in pred do
      if not i in set then 
        Add(set, i); 
        i.taken := true;
        set := getComponent(dag, set, i);
      fi;
    od;
    for i in succ do
      if not i in set then 
        Add(set, i); 
        i.taken := true;
        set := getComponent(dag, set, i);   
      fi;
    od;
    return set;
  end;

  answer := [];
  for i in dag do
    i.taken := false;
  od;
 
  while not Length(Filtered(dag, x -> not x.taken)) = 0  do
    set := Filtered(dag, x -> not x.taken);
    temp := getComponent(dag, [], set[1]);
    Add(answer, temp);
  od;
   
  if Length(answer) = 1 then
    return rec(graph := answer[1], modified := false);
  else
    return rec(graph := answer, modified := true);
  fi;
end; 

###############################################################################
## 
#F  BreakDeps ( <dag> )  . . . . . . . . .  breaking one-to-many dependencies
##

BreakDeps := function (dag)
  local subset, split, graph, i;
  subset := List([1..Length(dag)], i -> rec( 
    node_index := i,
    pred := Filtered(List(dag[i].pred, e->Position(dag, e)), e->e <> false),
    succ := Filtered(List(dag[i].succ, e->Position(dag, e)), e->e <> false)));
  split := Filtered(subset, r -> let( max := Maximum0(List(r.succ, i->Length(subset[i].succ))),
             Length(r.pred)=0 and Length(r.succ) > 0 and max > 0 and Length(r.succ) > 2*max));
  split := List(split, e->e.node_index);
  if Length(split) > 0 then
    graph := [[],[]];
    for i in [1..Length(dag)] do
      if i in split then
        Add(graph[1], dag[i]);
      else
        Add(graph[2], dag[i]);
      fi;
    od;
    return rec(graph := graph, modified := true);
  else
    return rec(graph := dag, modified := false);
  fi;
end;

###############################################################################
## 
#F  PartitionDag ( <dag> )  . . . . . . . . . . . . returns a partitioned dag 
##
PartitionDag := function(dag)
  local hasBlackNode, bBlackNodes, i, redlist, bluelist, serial;

  serial := BreakDeps(dag);
  if serial.modified then
    return List(serial.graph, x->PartitionDag(x));
  else 
    serial := SplitDag(dag);
    if serial.modified then
      return List(serial.graph, x->PartitionDag(x));
    else
    
    # Parallel Split
    if Length(dag)>=2 then
      
      InitColor(dag);
      bBlackNodes := false;
      while (Length(Filtered(dag, x -> x.color = BLACK))>0) do
        bBlackNodes := true;
        ColorPhase(dag, RED);
        ColorPhase(dag, BLUE);     
      od;
  
      if bBlackNodes then
        redlist := Filtered(dag, x->x.color=RED);
        bluelist := Filtered(dag, x->x.color=BLUE);
        return [PartitionDag(redlist), PartitionDag(bluelist)];
      fi;
    fi; 
    return dag;
    fi;
  fi;
end;

##########################################################################
##
#F  ScheduleDag( <partitioned_dag> ) . . . . schedules a paritioned dag
##
ScheduleDag := function(pdag)
  local f, newlist, bigorder; 

  newlist := [];
  bigorder := 0;
 
  f := function(pdag)
    local i,j,k,m;     
    if Length(pdag)>0 then
      if IsRec(pdag[1]) then          # use bfs for the order of the nodes        
        for j in pdag do j.taken := false; j.order:=0; od;
        while Length(Filtered(pdag, x->not x.taken))>0 do
          k := Filtered(pdag, x -> not x.taken);
          # want all nodes not taken, with preds either outside pdag or taken
          for j in k do
            m := true;
            for i in j.pred do
              if (i in pdag) and (not i.taken) then
                m := false;
              fi;
            od; 
            if m then 
               j.taken := true; 
               Add(newlist, j.cmd); 
               bigorder := bigorder + 1;
               j.order := bigorder;  
            fi;
          od;
        od;
      elif IsList(pdag[1]) then
        for i in [1..Length(pdag)] do
           f(pdag[i]);
        od;        
      fi;
    fi; 
  end;

  f(pdag);
  return newlist;
  
end;



###############################################################################
##  
#F  RandomScheduleDag( <dag> ) . . . . . . schedules a dag in a random manner
##
RandomScheduleDag := function(dag)
  local f, newlist, bigorder, list_of_recs; 

  newlist := [];
  bigorder := 0;
 
  f := function(dag)
    local i,j,k,m;     
    if Length(dag)>0 then
      if IsRec(dag[1]) then         
        for j in dag do j.taken := false; j.order:=0; od;
        while Length(Filtered(dag, x->not x.taken))>0 do
          k := Filtered(dag, x -> not x.taken);
          list_of_recs := [];   
          # want all nodes not taken, with preds either outside dag or taken
          for j in k do
            m := true;
            for i in j.pred do
              if (not i.taken) then
                m := false;
              fi;
            od; 
            if m then 
              Add(list_of_recs, j);
            fi;
          od;   
          j := Random(list_of_recs);
          j.taken := true; 
          Add(newlist, j.cmd); 
          bigorder := bigorder + 1;
          j.order := bigorder;  
        od;
      fi;
    fi; 
  end;

  f(dag);
  return chain(Filtered(newlist, x -> x <> 0));
  
end;



###############################################################################
##
#F  GetCompleteSchedules( <dag>, <int_limit> ) . . . . . get complete set of 
#F      schedules of a given dag, with the number of limit of the number of 
#F      schedules
##
GetCompleteSchedules := function(dag, limit)
  local f, newlist, bigorder, li, dag_length, j; 

  newlist := [];
  dag_length := Length(dag);
 
  f := function(dag, li, bigorder)
    local i,j,k,m, sel, newdag, li2;

    if Length(newlist) >= limit then
      return;
    fi; 
      
    if Length(li) = dag_length then
      m := Filtered(li, x -> x<>0);
      if Length(Filtered(newlist, x->x=m))<=0 then
        Add(newlist, m);
      fi;
    elif Length(dag)>0 then
      if IsRec(dag[1]) then          # use bfs for the order of the nodes        
        newdag := Copy(dag);          
        # want all nodes not taken, with preds either outside dag or taken
        for j in [1..Length(dag)] do
          if not dag[j].taken  then
            m := true;
            for i in dag[j].pred do
              if (not i.taken) then
                m := false;
              fi;
            od; 
            if m then
              dag[j].taken := true; 
              li2 := Copy(li);
              if dag[j].cmd <> 0 then
                Add(li2, j); 
              else 
                Add(li2, 0);
              fi;
              dag[j].order := bigorder;  
              newdag := Copy(dag);
              f(newdag, li2, bigorder + 1); 
              dag[j].order := 0;
              dag[j].taken := false;          
            fi;
          fi;
        od;         
      fi;
    fi; 
  end;

  for j in dag do j.taken := false; j.order:=0; od;
  f(dag, [], 0);

  newlist := List(newlist, x -> chain(List(x, y -> dag[y].cmd)));

  return newlist;  
end;






###############################################################################
##
#F  FFTWScheduleAssignments( <chain> ) . . . schedule assignments 
#F      using FFTW's scheduler
##
FFTWScheduleAssignments := function(chain_obj)
  local dag, pdag, schedule;
  
  dag := BuildDag(chain_obj);
  pdag := PartitionDag(dag); #SplitDagByDataType(dag);

  schedule := chain(ScheduleDag(pdag));

  if PRINTDAG then   
    PrintTo("SA_dag", PrintDag(dag, false));
    SYS_EXEC("dot -Tps -odag.ps SA_dag");
    SYS_EXEC("rm SA_dag");

    PrintTo("SA_dag", PrintDag(dag, true));
    SYS_EXEC("dot -Tps -oorder.ps SA_dag");
    SYS_EXEC("rm SA_dag");

    PrintTo("SA_pdag", PrintDag(pdag, false));
    SYS_EXEC("dot -Tps -opdag.ps SA_pdag");
    SYS_EXEC("rm SA_pdag");

    PrintTo("SA_oldschedule", chain_obj);
    PrintTo("SA_newschedule", schedule);  

    PrintTo("SA_order", PrintDag(pdag, true));
    SYS_EXEC("dot -Tps -oporder.ps SA_order");
    SYS_EXEC("rm SA_order");
  fi;

  return schedule; 
end;

###############################################################################
##
#F  RandomScheduleAssignments(<chain>, <int>) . . . generates a random schedule
##
RandomScheduleAssignments := function(chain_obj)
  local dag, schedule;
  
  dag := BuildDag(chain_obj);

  schedule := RandomScheduleDag(dag);

  if PRINTDAG then 
    PrintTo("SA_dag", PrintDag(dag, false));
    SYS_EXEC("dot -Tps -ordag.ps SA_dag");
    SYS_EXEC("rm SA_dag");

    PrintTo("SA_oldschedule", chain_obj);
    PrintTo("SA_newschedule", schedule);  

    PrintTo("SA_dag", PrintDag(dag, true));
    SYS_EXEC("dot -Tps -ororder.ps SA_dag");
    SYS_EXEC("rm SA_dag");

  fi;

  return schedule; 
end;

###############################################################################
##
#F  CompleteScheduleAssignments (<chain>, <int_limit>) . . . 
#F      generates all the schedules (with limit as the number of schedules 
#F      generated)
##
CompleteScheduleAssignments := function(chain_obj, number)
  local dag;
  
  dag := BuildDag(chain_obj);
  return GetCompleteSchedules(dag, number);

end;


###############################################################################
##
#F  EdwardScheduleAssignments( <chain>, <int_reg>) . . . using
#F      Edward's scheduling algorithm, generate a schedule
##
EdwardScheduleAssignments := function(chain_obj, n)
  local dag,i,j,k,l,m,mincost,minindex,readylist, final, registers, getcost, costlist;
  
  dag := BuildDag(chain_obj);

  # init readylist and taken
  readylist := Filtered(dag, x -> x.pred=[]);    
  for i in dag do
    i.taken := false;
  od;
  
  
  final := [];
  registers := List([1..n], x->1);

  getcost := function(node)
    local i,j;
    i:=0;
    for j in node.pred do
      if not (i in registers) then
        i:=i+2; 
      else
        i:=i-1;
      fi;
    od;
    return i;  
  end;

  while (Length(readylist)>0) do
     # for each node in readylist, determine which to pick 
     #   among the nodes in the readylist, pick one with   
     #   least cost
     #   1) determine cost
     #   2) determine min
     #   3) take min
     #
     #   N.B. maintain registers
     costlist := List(readylist, x->getcost(x));
     minindex := 1;
     mincost := costlist[1];
#     Print(costlist, "\n");
#     Print("reg = ", Length(Filtered(registers, x->x<>1)), "\n");

     for i in [1..Length(costlist)] do 
       if mincost > costlist[i] then
         minindex := i;
         mincost := costlist[i];
       fi; 
     od; 

     k := readylist[i];
#     Print(k.original, "\n");

     # update registers
     if k in registers then
       i := Position(registers, k);
       j := [k];
       for l in [1..i-1] do
         Add(j, registers[l]);  
       od; 
       for l in [i+1..n] do
         Add(j, registers[l]);  
       od; 
       registers := j;
     else
       j := [k];
       for l in [1..n-1] do
         Add(j, registers[l]);  
       od; 
       registers := j; 
     fi; 

     readylist := Filtered(readylist, x -> not x=k);
     Add(final, k.cmd);
     k.taken := true;

     for i in k.succ do
       if not i.taken then 
         l := true;
         for j in i.pred do
           if not j.taken then
             l := false;
           fi;
         od;
         if l then  
           Add(readylist, i);
         fi;
       fi;
     od;


  od;
  

  return chain(Filtered(final, x->x<>0));  

end;

###############################################################################
##
#F  ImFFTWScheduleAssignments( <chain> ) . . . using an algorithm
#F      similar to Edwards scheduler, generate a schedule
##
ImFFTWScheduleAssignments := function(chain_obj)
  local dag, first_level, second_level, output_nodes,i,j,k, color, color_no, li;
  
  dag := BuildDag(chain_obj);

  # label second level nodes
  for i in dag do
    i.level := 3;        # OTHER LEVEL
  od;
  first_level := Filtered(dag, x -> x.pred=[]);    
  second_level := [];
  for i in first_level do
    i.level:= 1;         
    for j in i.succ do
      j.level:= 2;
      if not (j in second_level) then
        Add(second_level, j);
      fi;
    od;
  od;     

  # get outputnodes;
  output_nodes := Filtered(dag, x->x.succ = []);
 
  # color output nodes
  for i in dag do
    i.color := 0;  # means uninitialized
  od;
  color := function(node, c)
    local i;
    if not node.color = 0 then
      return;
    elif node.level=3 then
      node.color := c;
      for i in node.pred do
        color(i, c);
      od;
      for i in node.succ do
        color(i, c);
      od;
    elif node.level = 2 then
      if node.color = 0 then
        node.color := c;
      fi;
    fi;
  end;

  color_no := 0;
  for i in output_nodes do
    if i.color=0 then
      color_no := color_no + 1;
      color(i, color_no);
    fi;
  od;  
  

  # for each color, schedule graph
  li := [];
  for i in [1..color_no] do
    j := Filtered(dag, x -> x.color=i);
    j := ScheduleDag(j);    
    for k in [1..Length(j)] do
      Add(li, j[k]);
    od;
  od;
  

  # output statements;
  return chain(li);  
end;


# tochain := x -> Cond(IsChain(x), Copy(x), chain(Copy(x)));

# bsplit := function(code)
#     local dims;
#     dims := code.dimensions;
#     code := When(IsBound(code.cmd), code.cmd, code);
#     code := FlattenCode(BinSplit(tochain(code)));
#     code := Compile.declareVars(code);
#     code.dimensions := dims;
#     return code;
# end;

# sched := function(code)
#     local dims;
#     dims := code.dimensions;
#     code := When(IsBound(code.cmd), code.cmd, code);
#     code := FFTWScheduleAssignments(tochain(code));
#     code := Compile.declareVars(code);
#     code.dimensions := dims;
#     return code;
# end;

# sched2 := function(code)
#     local dims;
#     dims := code.dimensions;
#     code := When(IsBound(code.cmd), code.cmd, code);
#     code := EdwardScheduleAssignments(tochain(code), 8);
#     code := Compile.declareVars(code);
#     code.dimensions := dims;
#     return code;
# end;

# randsched := function(code)
#     local dims;
#     dims := code.dimensions;
#     code := When(IsBound(code.cmd), code.cmd, code);
#     code := RandomScheduleAssignments(tochain(code));
#     code := Compile.declareVars(code);
#     code.dimensions := dims;
#     return code;
# end;



