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
