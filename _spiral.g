if not IsBound(LocalConfig) then
	#Print("\nLoading spiral\n");
	Load(spiral);
fi;

LoadPackage("arep");

LocalConfig.cpuinfo := CopyFields(SupportedCPUs.Core_AVX, rec(
    freq := 2195
));

LocalConfig.osinfo := SupportedOSs.Linux64;

SpiralDefaults.profile := spiral.profiler.default_profiles.linux_x86_gcc;

LocalConfig.compilerinfo := SupportedCompilers.GnuC;

