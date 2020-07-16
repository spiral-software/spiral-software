
# Copyright (c) 2018-2020, Carnegie Mellon University
# See LICENSE for details


RequirePackage("arep");

Package(spiral);

Include(config);
Include(trace);

Load(spiral.rewrite);
Load(spiral.code);     #ProtectNamespace(code);
Load(spiral.approx);
Load(spiral.spl);      #ProtectNamespace(spl);
Load(spiral.formgen);  #ProtectNamespace(formgen);
Load(spiral.sigma);

Declare(CMeasure);
Declare(CMatrix);

Load(spiral.compiler);
Global.compiler := spiral.compiler;

Load(spiral.fpgen);


Include(defaults);
Include(perfstat);

Load(spiral.search);

Load(spiral.transforms);
Global.transforms := spiral.transforms;

Load(spiral.paradigms);

Load(spiral.transforms.filtering);

Load(spiral.profiler);
CMeasure := spiral.profiler.CMeasure;
CMatrix  := spiral.profiler.CMatrix;

Load(spiral.nontransforms);

Load(spiral.platforms);

Load(spiral.libgen);

Include(prdft_defaults);

Include(cpus);
Include(compilers);
Include(test);

Load(spiral.sym);

Load(spiral.scriptgen);

Include(bench);
Include(debug);


spiral.rewrite := rewrite;
spiral.code := code;
spiral.approx := approx;
spiral.spl := spl;
spiral.formgen := formgen;
spiral.sigma := sigma;
spiral.compiler := compiler;
spiral.libgen := libgen;
#spiral.web := web;
spiral.platforms := platforms;
spiral.paradigms := paradigms;
spiral.profiler := profiler;
spiral.sym := sym;
spiral.transforms := transforms;

####
# NOTE: hack, should be looking up Localconfig.compiler...

if LocalConfig.osinfo.isDarwin() then
    SpiralDefaults.profile := spiral.profiler.default_profiles.darwin_x86_icc; fi;
if LocalConfig.osinfo.isLinux() then
    SpiralDefaults.profile := spiral.profiler.default_profiles.linux_x86_icc; fi;

Add(HooksSessionStart, function()
    Import(spl, formgen, code, rewrite,
       transforms, transforms.dft, search, compiler, sigma, platforms, platforms.sse, platforms.avx,
       paradigms.common, paradigms.vector, paradigms.cache, libgen, profiler,
       sym, platforms.intel);
    LocalConfig.info();
    Print("\n");
end);


NamespaceAdd(Global, spiral);
