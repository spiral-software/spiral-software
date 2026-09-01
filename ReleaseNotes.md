## Release Notes for Spiral Version 8.6.0

### Introduction

These release notes for Spiral 8.6.0 provide an overview of the release and
document any known issues.  For details of the changes applied since the last
release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for
a specific platform.

### Get Spiral Version 8.6.0

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

#### Dynamic CUDA Architecture Detection:
Added `SpiralCUDAUtils.cmake` to automatically detect local GPU compute
capabilities using `nvidia-smi`. Includes a version-safe fallback strategy
(targeting Ampere/`sm_80`, Hopper/`sm_90` on CUDA 11.8+, and Blackwell/`sm_120`
on CUDA 12.8/13.0+) for headless environments such as HPC login nodes or WSL.

#### **GAP Plugin Architecture & C Extensions:**
 * Introduced native dynamic plugin loading support via `LoadPlugin` in GAP
   (`plugins.c`), complete with cross-platform handling (`dlopen`/`dlsym` on
   Unix/Linux, native `win_dlfcn` fallback on Windows). 
 * Added `EvalString` capability to execute GAP statements directly from string buffers.
 * Added `rec2json.g` package utility for serializing GAP records and lists to JSON strings.

#### **CMake & Toolchain Updates:**
 * Enhanced CUDA toolchain validation in `support/CMakeLists.txt` using
   `check_language(CUDA)` to gracefully fall back to CPU mode if a compiler or
   active driver is missing or mismatched. 
 * Replaced deprecated `helper_cuda.h` dependency in GPU detection scripts with standard `<cuda_runtime.h>` calls.
 * Ensured `SPIRAL_HOME` is dynamically resolved and normalized in CMake profiler target configurations.

#### **Profiler Enhancements:**
* **Slurm / HPC Job Execution:** Updated `localprofiler.py` to check for active `SLURM_JOB_ID`
  environments before attempting batch job submission, preventing redundant
  sub-job launches when already inside an allocated Slurm node. 
* **Test Runner Improvements:** Added support for `opts.profile.debug` and `opts.profile.keeptemp`
  in basic profiler tests.  `debug` forces the profiler to behave as if `--debug`
  (or -D) was passed and enables more verbose logging to `stdout`.  `keeptemp`
  causes the profiler to keep temporary directories and build files in the event
  the test fails, allowing easier inspection.

#### General Cleanup
 * Upgraded GitHub Actions workflow to use current versions (e.g., checkout).
 * Updated Ubuntu CI workflows (`ubuntu.yml`) to explicitly set `SPIRAL_HOME` and improve test job execution.

### Bug Fixes
* **Issue [#144](https://github.com/spiral-software/spiral-software/issues/144) Resolved:** Replaced
  deprecated `<termio.h>` header with `<termios.h>` to resolve build failures on
  newer Linux distributions (e.g., Ubuntu 26.04 / `ubuntu-resolute`). 

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./LICENSE) file for the full text). 

----------------------------------------------------------------------------------------------------

## Release Notes for Spiral Version 8.5.3

### Introduction

These release notes for Spiral 8.5.3 provide an overview of the release and
document any known issues.  For details of the changes applied since the last
release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for
a specific platform.

### Get Spiral Version 8.5.3

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

No major new features added, this is a maintenance release.

#### General Cleanup
* Fix stricter type checking errors for newer compilers (gcc-14 / icx)
* Cleanup and simplify CUDA build (only basic runtime needed)
* Enhancement to profiler to submit batch [SLURM] jobs when available
* Added option to specify a named profiler build directory
* Added ability to set [more] options for profiler from within Spiral (see profiler README)

### Bug Fixes
* Changes to the profiler to resolve issues with choice of float or double
* Fixed profiler functions to correctly handle float/double types with CUDA

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./LICENSE) file for the full text). 

----------------------------------------------------------------------------------------------------

## Release Notes for Spiral Version 8.4.1

### Introduction

These release notes for Spiral 8.4.1 provide an overview of the release and
document any known issues.  For details of the changes applied since the last
release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for
a specific platform.

### Get Spiral Version 8.4.1

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

No major new features added, this is a maintenance release.

#### General Cleanup
* Added a '-B' option that cause SPIRAL to print its build information and exit
* Cleanup of some old code
* Tracing features improved and cleaned up 

### Bug Fixes

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./LICENSE) file for the full text). 

----------------------------------------------------------------------------------------------------

## Release Notes for Spiral Version 8.4.0

### Introduction

These release notes for Spiral 8.4.0 provide an overview of the release and
document any known issues.  For details of the changes applied since the last
release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for
a specific platform.

### Get Spiral Version 8.4.0

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

#### General Cleanup
* Numerous changes were made to clean up some old code, simplifying (removing
when possible), relying on standard library functions, and providing a single
method to perform a function instead of supporting multiple methods.
* Reduced and simplified the number of environment variables that were (or could
be) set to control Spiral's operation. 
* Removed [old, deprecated, or unused] code.

#### CUDA
Several examples (also doubling as tests) added to demonstrate CUDA code generation and testing.

#### Profiler
Several enhancements to improve the performance of the profiler.
Improved handling of matrices returned by profiler, added ability to specify
upper/lower corners of a sub-matrix (test or verify a portion of a large
matrix). 
Get a list of the best timed ruletrees
In addition, cmake is now the standard build tools across all platforms when
profiling Spiral generated code. 

### Bug Fixes

* Fixed / improved several minor issues for profiler

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./LICENSE) file for the full text). 

----------------------------------------------------------------------------------------------------

## Release Notes for Spiral Version 8.3.0

### Introduction

These release notes for Spiral 8.3.0 provide an overview of the release and document the known issues.  For details of the changes applied since the last release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get Spiral Version 8.3.0

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

#### Spiral Packages:
Automatically search for and include in the build process any properly installed SPIRAL
package (i.e., packages added to <spiral>/namespaces/packages/XXXX).  Each package must
have its own **CMakeLists.txt** file.  Any tests included with the package may then be run
as part of Spiral testing.

#### CUDA:
First release to provide support for CUDA and generating GPU code. 
Added support for CUDA compiler (nvcc).
Added support and revised breakdown rules to provide CUDA support.

#### Spiral Code Generation:
Some tweaks and minor enhancements, including the ability to output the ruletree used when
outputting/printing code

#### BuildInfo:
A new Spiral command: BuildInfo(), can be run at the Spiral prompt to get information
about the build (e.g., version, branch, etc).  Output from this should be included with
any issue(s) reported.

### Bug Fixes

* Fixed couple of minor issues with PROFILER_LOCAL_ARGS (starting profiler)
* Remove "load()", in favor of using "Load()" exclusively (obsolete/redundant)
* Improved comments / removed obsolete/dead code
* Avoid AppendTo() (just Print()) to speed up writing of large generated files
* Fix target for profiler when on PPC
* Fix rows/columns mismatch in profiler for CVector
* Fix (add CVector) support in profiler on PPC

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD License (see the [**LICENSE**](./LICENSE) file for the full text).



## Release Notes for Spiral Version 8.2.0

### Introduction

These release notes for Spiral 8.2.0 provide an overview of the release and document the known issues.  For details of the changes applied since the last release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get Spiral Version 8.2.0

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

#### Memory Management:
The memory manager has been significantly overhauled.  SPIRAL will allocate
multiple memory Arenas to handle storage requirements, limited by the resources
to OS is willing to grant.  The initial amount of memory allocated for data
storage (default is 1 GByte) is used as the size for additional arenas.

#### CVector:
Added new profiler target request CVector(code, vector, opts) that applies the transform (implemented by code) to the vector and returns the result.

#### Testing:
New tests added and more rigourous checking to ensure test results are valid.
Report if a test is skipped (prior to this skipped was treated as passed).

Support building 32 bit version (with cmake) on Windows

### Bug Fixes

* Rewrote internal GAP function ProdVectorMatrix() to properly support symbolic matrix-matrix multiply
* Fixed input buffer switching problem in lexer that manifested with ReadValFromFile()
* Removed obsolete/dead code
* Eliminated many functions in favor of standard library calls that do the same thing 
* Fixed targets to include -fopenmp to enable Open MP test programs to link

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD License (see the [**LICENSE**](./LICENSE) file for the full text).



## Release Notes for Spiral Version 8.1.2

### Introduction

These release notes for Spiral 8.1.2 provide an overview of the release and document the known issues.  For details of the changes applied since the last release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get Spiral Version 8.1.2

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

No significant new features, this is a maintenance release.

### Bug Fixes

* Fixed sums_ruletree bug demonstrated by new test Advanced/DFT_PD_Stage1.g
* Added target win-x86-llvm (for LLVM compiler) to profiler targets
* Removed obsolete/dead code

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD License (see the [**LICENSE**](./LICENSE) file for the full text).



## Release Notes for Spiral Version 8.1.1

### Introduction

These release notes for Spiral 8.1.1 provide an overview of the release and document the known issues.  For details of the changes applied since the last release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get Spiral Version 8.1.1

You can download the latest release from:

https://github.com/spiral-software/spiral-software.git

## Change Summary

### New Features

No significant new features, this is a maintenance release.

### Bug Fixes

* Fixed arbitrary precision integer values for large integers (> 2^60 / 2^28 on 64/32 bit processsors).
* Fixed obscure memory (data) overwrite leading to random crash.
* Exit with fatal error and message if run out of memory.
* Added test for large/small integer arithmetic.
* Minor fixes/cleanup in code generation phase.

### Known Issues

None at present.

## License

Spiral is open source software licensed under the terms of the Simplified BSD License (see the [**LICENSE**](./LICENSE) file for the full text).
