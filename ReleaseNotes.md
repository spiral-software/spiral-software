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
