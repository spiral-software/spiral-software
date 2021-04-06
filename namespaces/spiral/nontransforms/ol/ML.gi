
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(Drawer, rec(
    __call__ := meth(self,n)
        return WithBases(self,rec(n:=n,instances:=[]));
    end,
    
    add := meth(self,instance)
        if (not IsList(instance)) or (Length(instance)<>self.n) then
            Error("Trying to add a non regular instance");
        fi;
        Add(self.instances,instance);
    end,

    getNormalTransfo := meth(self)
       local result,l,m,s;
       result:=[];
       for l in Transposed(self.instances) do
          m:=Mean(l);
          s:=Try(StdDev(l));
          if not s[1] then s:=0; else s:=s[2]; fi;
          Add(result,[m,s]);
       od;
       return result;
    end,

    normalize := meth(self,transfo)
       local t,n,l;
       t:=[];
       n:=1;
       for l in Transposed(self.instances) do
          if (transfo[n][2]<>0) then
              Add(t,List(l,x->(x-transfo[n][1])/transfo[n][2]));
          else
              Add(t,List(l,x->(x-transfo[n][1])));
          fi;
          n:=n+1;
       od;
       self.instances:=Transposed(t);
    end,

    checkCredibility := meth(self,testing, transfo) 
        local j;
        for j in [1..Length(self.instances)] do
            for i in [1..Length(testing.instances[1])] do
               if transfo[i][2]=0 and transfo[i][1]<>testing.instances[j][i] then
                   self.instances[j]:=[2^16,1,2^32];
               fi;
            od;
        od;
    end,

    denormalize := meth(self,transfo)
       local t,n,l;
       t:=[];
       n:=1;
       for l in Transposed(self.instances) do
          if (transfo[n][2]<>0) then
              Add(t,List(l,x->x*transfo[n][2] + transfo[n][1]));
          else
              Add(t,List(l,x->x + transfo[n][1]));
          fi;
          n:=n+1;
       od;
       self.instances:=Transposed(t);
    end,

    print := meth(self,filename)
       local l,a;
       PrintTo(filename,"");
       for l in self.instances do
          for a in l do
             AppendTo(filename,a," ");
          od;
          AppendTo(filename,"\n");
       od;
    end,

));

Class(GaussianPredictor, rec(

    __call__ := meth(self)
        return WithBases(self);
    end,

    predict := meth(self, hash, objid, testCases, tags)
      local n,a,i,transfo,training,testing,predict,index,sol,t;
      n:=DimensionsMat(testCases)[2];
      training:=Drawer(n+1);
      for a in Flat(hash.entries) do
          if ObjId(a.key)=objid and a.key.params[n+1]=tags and Length(a.data)>0 and IsBound(a.data[1].mflopslibgen) then
              training.add(Concat(a.key.params{[1..n]},
                      [a.data[1].mflopslibgen]));
          fi;
      od;

      transfo:=training.getNormalTransfo();
      a:=Copy(training);
      a.normalize(transfo);
      a.print("trainingset");

      testing:=Drawer(n);
      testing.instances:=testCases;
      a:=Copy(testing);
      a.normalize(DropLast(transfo,1));
      a.print("testingset");

      IntExec("rm -f rlog.gp");
      IntExec(Concat("gp-spec rlog.gp ",String(n),
              " 1 10 / 0.05:0.4 0.05:0.4"));
      IntExec("model-spec rlog.gp real 0.05:0.4");
      IntExec(Concat("data-spec rlog.gp ",String(n)," 1 / trainingset ."));
      IntExec("gp-gen rlog.gp fix 0.5 0.1");
      IntExec("mc-spec rlog.gp heatbath hybrid 18:3 0.5");
      IntExec("gp-mc rlog.gp 100");
      IntExec("gp-pred nqb rlog.gp 51: / testingset >myoutput");
      IntExec("awk 'BEGIN {print \"[\"} {print \"[\",$1,\", \",$2,\", \",$3,\"], \"} END{print\"];\"}' myoutput >myrealoutput");

      a:=ReadVal("myrealoutput");

      predict:=Drawer(3);
      predict.instances:=a;
      predict.denormalize(Replicate(3,Last(transfo)));

      for a in [1..Length(testing.instances)] do
          t:=ApplyFunc(objid,Concat(testing.instances[a],[tags]));
          i:=HashLookup(hash,t);
          if (i<>false) and IsBound(i[1].mflopslibgen) then
              predict.instances[a]:=Replicate(3,i[1].mflopslibgen);
          fi;
      od;

      return predict.instances;
   end,

    predict2 := meth(self, hash, objid, testCases, tags)
      local n,a,i,transfo,training,testing,predict,index,sol,t,tmp;
      n:=DimensionsMat(testCases)[2];
      training:=Drawer(n+1);
      for a in Flat(hash.entries) do
          if ObjId(a.key)=objid and a.key.params[n+1]=tags and Length(a.data)>0 and 
              IsBound(a.data[1].measured)  then
              training.add(Concat(a.key.params{[1..n]},
                      [_compute_mflops(When(IsBound(a.key.normalizedArithCost), 
                                  a.key.normalizedArithCost(), 0), a.data[1].measured)]));
          fi;
      od;
      Print("Estimating ",Length(testCases)," points (with ",Length(training.instances),"): ", 
          objid," (",tags,")\n");

      if Length(training.instances)=0 then
          return Replicate(Length(testCases),[2^16,0,2^32]);
      fi;
      transfo:=training.getNormalTransfo();
      a:=Copy(training);
      a.normalize(transfo);
      a.print("trainingset");

      testing:=Drawer(n);
      testing.instances:=testCases;
      a:=Copy(testing);
      a.normalize(DropLast(transfo,1));
      a.print("testingset");

      IntExec("rm -f rlog.gp");
      IntExec(Concat("gp-spec rlog.gp ",String(n),
              " 1 10 / 0.05:0.4 0.05:0.4"));
      IntExec("model-spec rlog.gp real 0.05:0.4");
      IntExec(Concat("data-spec rlog.gp ",String(n)," 1 / trainingset . 2>/dev/null"));
      IntExec("gp-gen rlog.gp fix 0.5 0.1");
      IntExec("mc-spec rlog.gp heatbath hybrid 18:3 0.5");
      IntExec("gp-mc rlog.gp 100");
      IntExec("gp-pred nqb rlog.gp 51: / testingset >myoutput");
      IntExec("awk 'BEGIN {print \"[\"} {print \"[\",$1,\", \",$2,\", \",$3,\"], \"} END{print\"];\"}' myoutput >myrealoutput");

      a:=ReadVal("myrealoutput");

      predict:=Drawer(3);
      predict.instances:=a;
      predict.denormalize(Replicate(3,Last(transfo)));

      predict.checkCredibility(testing, transfo);

      for a in [1..Length(testing.instances)] do
          t:=ApplyFunc(objid,Concat(testing.instances[a],[tags]));
          i:=HashLookup(hash,t);
          if (i<>false) and IsBound(i[1].measured) then
              predict.instances[a]:=Replicate(3,i[1].measured);
          else
              predict.instances[a]:=List(predict.instances[a],x->
                  _compute_cycles_from_mflops(When(IsBound(t.normalizedArithCost),
                                  t.normalizedArithCost(), 0),x));
              #invert max and min because min flops is max runtime
              tmp:=predict.instances[a][2];
              predict.instances[a][2]:=predict.instances[a][3];
              predict.instances[a][3]:=When(tmp<0,2^64,tmp);
          fi;
      od;

      return predict.instances;
   end,

 
   countTraining:=meth(self, dpbench, objid, n, tags)
      local r,hash,a;
      hash:=dpbench.exp.bases.hashTable;
      r:=0;
      for a in Flat(hash.entries) do
          if ObjId(a.key)=objid and a.key.params[n+1]=tags and Length(a.data)>0 then
              r:=r+1;
          fi;
       od;      
      return r;
   end,

   findMax:=meth(self, base_bench, libgen_bench, base_objid, libgen_objid, testing, tags,flopsfactor)
       local over,sol,secured,estimate,a,s, unscaledsol;
       base_bench.verbosity:=0;
       if (g.countTraining(base_bench,base_objid,DimensionsMat(testing)[2],tags)<2) then
           base_bench.exp.bases.benchTransforms:=[ApplyFunc(base_objid,Concat(First(testing,x->true),[tags])),[ApplyFunc(base_objid,Concat(Last(testing),[tags]))]];
           base_bench.resumeAll();
       fi;

       over:=false;
       while(not over) do
           unscaledsol:=self.predict(base_bench.exp.bases.hashTable,base_objid,testing,tags);
           sol:=List([1..Length(testing)],x->List(unscaledsol[x],y->flopsfactor(testing[x],y)));
           secured:=[testing[1],sol[1]];
           estimate:=[testing[1],sol[1]];
           for a in [1..Length(testing)] do
              s:=[testing[a],sol[a]];
              if (s[2][1]=s[2][2] and s[2][1]=s[2][3] and s[2][1]>secured[2][1]) then
                  secured:=s;
              fi;
              if (s[2][3]>estimate[2][3]) then
                  estimate:=s;
              fi;
           od;

           Print("Current Maximum is ",Red(StringDouble("%.3f",secured[2][1]/1000))," [Gf/s] achieved by ",ApplyFunc(base_objid,Concat(secured[1],[tags])),"\n" );
           if (estimate[2][3]<=secured[2][1]) then
               over:=true;
           else
               Print("Best Hope is ",ApplyFunc(base_objid, Concat(estimate[1],[tags]))," with ",StringDouble("%.3f",estimate[2][1]/1000)," [Gf/s] (",StringDouble("%.3f",estimate[2][2]/1000)," < x < ",StringDouble("%.3f",estimate[2][3]/1000),")\n" );
               base_bench.exp.bases.benchTransforms:=[ApplyFunc(base_objid,Concat(estimate[1],[tags]))];
               base_bench.resumeAll();
               libgen_bench.runExhaust([ApplyFunc(libgen_objid,Concat(estimate[1]*2,[ApplyFunc(base_objid,Concat(estimate[1],[tags]))]))]);
               base_bench.exp.bases.benchTransforms:=[];
               base_bench.resumeAll(); #Fire Callbacks
           fi;
       od;
       return ApplyFunc(base_objid,Concat(secured[1],[tags]));
    end,
));
