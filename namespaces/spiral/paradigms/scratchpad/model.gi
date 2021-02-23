
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Class(LinearModel, rec(
    __call__ := (self, k, d) >> CopyFields(self, rec(k := k, d := d)),
    d := 0,
    k := 1,
    eval := (self, x) >> self.d + self.k * x
));

Class(ExpModel, rec(
    __call__ := (self, alpha, tau) >> CopyFields(self, rec(alpha := alpha, tau := tau)),
    alpha := 0.1,
    tau := 5,
    eval := (self, x) >> (self.alpha + (1 - d_exp(-self.tau * x))) / (1+self.alpha)
));


Class(ScratchModel, rec(
    config := rec(
        loadModel := (numPkts, pktSize) -> numPkts * pktSize / (0.2 * ExpModel(0.1, 0.2).eval(pktSize)),
        storeModel := (numPkts, pktSize) -> 1.1 * numPkts * pktSize / (0.2 * ExpModel(0.1, 0.2).eval(pktSize)),
        computeModel := x->LinearModel(1.1, 5).eval(x)
    ),
    getKernels := (self, spl) >>  List(Collect(self.updateInfo(spl), LSKernel), i->i.info),
    updateInfo := (self, spl) >>  SubstTopDown(spl, [@(1, Compose), @(2, DMAScat), @(3, LSKernel), @(4, DMAGath)],
        (e,cx) -> @(2).val *
            LSKernel(@(3).val.child(1), rec(
                opcount := @(3).val.info.opcount,
                loadFunc := @(4).val.func,
                storeFunc := @(2).val.func,
                free := When(IsBound(cx.ISum), Set(Concat(List(cx.ISum, i->i.var))), Set([])))) *
            @(4).val),

    getPacketData := meth(self, func)
        local domain, pktSize, numPkts;

        domain := func.domain();
        pktSize := 1;
        if ObjId(func) = fTensor and ObjId(Last(func.children())) = fId then
            pktSize := Last(func.children()).domain();
        fi;
        numPkts := domain / pktSize;

        return [pktSize, numPkts];
    end,

    modelStat := meth(self, spl)
        local kernels, stat, numPkts, pktSize, kernel, iter, func, n,
            flops, bw, loadcycles, storecycles, cpucycles, cycles;

        kernels := self.getKernels(spl);
        stat := [];

        for kernel in kernels do
            flops := kernel.opcount;
            cpucycles := self.config.computeModel(flops);

            n := kernel.loadFunc.domain();
            [pktSize, numPkts] := self.getPacketData(kernel.loadFunc);
            loadcycles := self.config.loadModel(numPkts, pktSize);
            [pktSize, numPkts] := self.getPacketData(kernel.storeFunc);
            storecycles := self.config.storeModel(numPkts, pktSize);
            iter := Product(List(kernel.free, i->i.range));
            Add(stat, rec(
                iter := iter,
                kernelcycles := Maximum(loadcycles, storecycles, cpucycles),
                loadcycles := loadcycles,
                storecycles := storecycles,
                cpucycles := cpucycles,
                balance := cpucycles / Maximum(loadcycles, storecycles)
            ));

        od;
        return stat;
    end,

    modelRuntime := meth(self, spl)
        local stat, s, runtime, cycles;

        runtime := 0;
        stat := self.modelStat(spl);

        for s in stat do
            cycles := s.loadcycles + s.storecycles + s.iter * s.kernelcycles;
            runtime := runtime + cycles;
        od;
        return When(IsInt(runtime), runtime, IntDouble(runtime));
    end,

    measureFunction := (rt, opts) -> let(s := SumsRuleTree(rt, opts),  paradigms.scratchpad.ScratchModel.modelRuntime(s))
));
