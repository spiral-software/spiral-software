
# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details


Import(code); # EvalScalar

_color := (c, str) -> When(LocalConfig.osinfo.useColor(), c(str), str);

_compute_cycles_from_mflops := (flop, flops) -> IntDouble(Double(EvalScalar(fdiv(flop,flops)))*LocalConfig.cpuinfo.freq);
_compute_mflops := (flop, cycles) -> Cond(cycles<>0, IntDouble(Double(EvalScalar(fdiv(flop,cycles)))*LocalConfig.cpuinfo.freq), "#NA");
_compute_gflops := (flop, cycles) -> Cond(cycles<>0, StringDouble("%.3f", (EvalScalar(fdiv(flop,cycles))/1000.0)*LocalConfig.cpuinfo.freq), "#NA");
_compute_gflops_freq := (flop, cycles, freq) -> Cond(cycles<>0, StringDouble("%.3f", (EvalScalar(fdiv(flop,cycles))/1000.0)*freq), "#NA");

_seqPerfStats := function(file, n, cycles, search_time)
    PrintLine("n ", n, "  seq ", BlueStr(cycles), " [cyc]  search ", search_time, " [s]");
   AppendTo(file,
    PrintLine("n ", n, "  seq ", cycles, " [cyc]  search ", search_time, " [s]"));
end;

_seqPerfStatsMflops := function(file, t, flops, cycles, search_time)
    local realFlops;
    if ObjId(flops)=ListClass then realFlops := flops[2]; flops := flops[1]; else realFlops := 0; fi;
    PrintLine(t, "  ", _color(BlueStr, cycles), " [cyc]  ", _color(DarkYellowStr, _compute_mflops(flops, cycles)), " [Mf/s]  ",
     _compute_mflops(realFlops, cycles), " [Mf/s, real]  ", search_time, " [search, s]");
    AppendTo(file,
    PrintLine(t, "  ", cycles, " [cyc]  ", _compute_mflops(flops, cycles), " [Mf/s]  ",  _compute_mflops(realFlops, cycles), " [Mf/s, real]  ", search_time, " [search, s]"));
end;

_seqPerfStatsGflops := function(file, t, freq, flops, cycles, search_time)
    local realFlops;
	
    if ObjId(flops)=ListClass then realFlops := flops[2]; flops := flops[1]; else realFlops := 0; fi;	
	
    PrintLine(t, "  ", _color(BlueStr, cycles), " [cyc]  ", _color(DarkYellowStr, _compute_gflops_freq(flops, cycles, freq)), " [Gf/s]  ",
     _compute_gflops_freq(realFlops, cycles, freq), " [Gf/s, real]  ", search_time, " [search, s]");
    AppendTo(file,
    PrintLine(t, "  ", cycles, " [cyc]  ", _compute_gflops_freq(flops, cycles, freq), " [Gf/s]  ", _compute_gflops_freq(realFlops, cycles, freq), " [Gf/s, real]  ", search_time, " [search, s]"));
end;

_seqPerfStatsMflopsAcc := function(file, n, flops, cycles, search_time, acc)
    local realFlops;
    if ObjId(flops)=ListClass then realFlops := flops[2]; flops := flops[1]; else realFlops := 0; fi;
    PrintLine("n ", n, "  seq ", _color(BlueStr, cycles), " [cyc] ", _color(DarkYellowStr, _compute_mflops(flops, cycles)), " [Mflop/s] search ",
    _compute_mflops(realFlops, cycles), " [Mflop/s, real] search ", search_time, " [s] accuracy ", -acc, " [dig]");
    AppendTo(file,
    PrintLine("n ", n, "  seq ", cycles, " [cyc] ", _compute_mflops(flops, cycles), " [Mflop/s] search ", _compute_mflops(realFlops, cycles), " [Mflop/s, real] search ",
    search_time, " [s] accuracy ", -acc, " [dig]"));
end;

_seqPerfStatsGflopsAcc := function(file, t, flops, cycles, search_time, acc)
    local digits, realFlops;
    if ObjId(flops)=ListClass then realFlops := flops[2]; flops := flops[1]; else realFlops := 0; fi;
    digits := When(IsFloat(acc), -IntDouble(d_log(1e-16+AbsFloat(acc))/d_log(10)), acc);
    PrintLine(t, "  ", _color(BlueStr, cycles), " [cyc]  ", _color(DarkYellowStr,
    _compute_gflops(flops, cycles)), " [Gf/s]  ", _compute_gflops(realFlops, cycles), " [Gf/s, real]  ", search_time, " [search, s] ", digits, " [dig]");
    AppendTo(file,
    PrintLine(t, "  ", cycles, " [cyc]  ", _compute_gflops(flops, cycles), " [Gf/s]  ", _compute_gflops(realFlops, cycles), " [Gf/s, real]  ",
    search_time, " [search, s] ", digits, " [dig]"));
end;

_parPerfStats := function(file, n, threads, cycles, search_time)
    PrintLine("n ", n, "  threads ", threads, "  par ", _color(BlueStr, cycles), " [cyc]  search ", search_time, " [s]");
   AppendTo(file,
    PrintLine("n ", n, "  threads ", threads, "  par ", cycles, " [cyc]  search ", search_time, " [s]"));
end;

printcount := function(opcount, countrec)
    local n, i, retval;
    retval := "";
    n := 1;
    for i in opcount do
      retval := ConcatenationString(retval, " ", String(i), " ", countrec.printstrings[n]);
      n := n + 1;
    od;
    return(retval);
end;

_seqPerfStatsGflopsCount := function(file, t, flops, cycles, search_time, count, countrec)
    local realFlops;
    if ObjId(flops)=ListClass then realFlops := flops[2]; flops := flops[1]; else realFlops := 0; fi;

    PrintLine(t, "  ", _color(BlueStr, cycles), " [cyc]  ", _color(DarkYellowStr, _compute_gflops(flops, cycles)), " [Gf/s]  ",
     _compute_gflops(realFlops, cycles), " [Gf/s, real]  ", search_time, " [search, s] ", printcount(count, countrec));

    AppendTo(file,
    PrintLine(t, "  ", cycles, " [cyc]  ", _compute_gflops(flops, cycles), " [Gf/s]  ",
     _compute_gflops(realFlops, cycles), " [Gf/s, real]  ", search_time, " [search, s] ", printcount(count, countrec)) );
end;

_seqPerfStatsGflopsAccCount := function(file, t, flops, cycles, search_time, acc, count, countrec)
    local digits, realFlops;
    if ObjId(flops)=ListClass then realFlops := flops[2]; flops := flops[1]; else realFlops := 0; fi;
    digits := When(IsFloat(acc), -IntDouble(d_log(1e-16+AbsFloat(acc))/d_log(10)), acc);

    PrintLine(t, "  ", _color(BlueStr, cycles), " [cyc]  ", _color(DarkYellowStr, _compute_gflops(flops, cycles)), " [Gf/s]  ",
     _compute_gflops(realFlops, cycles), " [Gf/s, real]  ", search_time, " [search, s] ", digits, " [dig]", printcount(count, countrec));

    AppendTo(file,
    PrintLine(t, "  ", cycles, " [cyc]  ", _compute_gflops(flops, cycles), " [Gf/s]  ",
     _compute_gflops(realFlops, cycles), " [Gf/s, real]  ", search_time, " [search, s] ", digits, " [dig]", printcount(count, countrec)) );
end;
