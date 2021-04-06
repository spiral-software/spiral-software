## Release Notes for Spiral Version 8.3.0

### Introduction

These release notes for Spiral 8.3.0 provide an overview of the release and document the known issues.  For details of the changes applied since the last release, please see the **Change Summary** below.

### Supported Platforms

Spiral is supported on Windows, Linux, and MacOS.

Spiral is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get Spiral Version 8.3.0

You can download the lastest release from:

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

You can download the lastest release from:

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

You can download the lastest release from:

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

You can download the lastest release from:

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
