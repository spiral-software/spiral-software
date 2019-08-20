if not IsBound(LocalConfig) then
	#Print("\nLoading spiral\n");
	Load(spiral);
fi;

LoadPackage("arep");

LocalConfig.cpuinfo := CopyFields(SupportedCPUs.Core_AVX, rec(
    freq := 2195
));

LocalConfig.osinfo := SupportedOSs.Windows8;

# set to true to use ICC or GCC, otherwise default to Visual Studio
UseICC := false;
UseGCC := false;
if (UseICC) then
	# ICC
	SpiralDefaults.profile := spiral.profiler.default_profiles.win_x64_icc;
	SpiralDefaults.target := rec(name:="win-x64-icc");
	LocalConfig.compilerinfo := SupportedCompilers.IntelC;
elif (UseGCC) then
	# GCC
	SpiralDefaults.profile := spiral.profiler.default_profiles.win_x86_gcc;
	SpiralDefaults.target := rec(name:="win-x86-gcc");
	LocalConfig.compilerinfo := SupportedCompilers.GnuC;
else
	# Visual Studio
	SpiralDefaults.profile := spiral.profiler.default_profiles.win_x86_vcc;
	SpiralDefaults.target := rec(name:="win-x86-vcc");
	LocalConfig.compilerinfo := SupportedCompilers.VisualC;
fi;



