# xSDK Community Policy Compatibility for SPIRAL

This document summarizes the efforts of current and future xSDK member packages to achieve compatibility with the xSDK community policies. Below only short descriptions of each policy are provided. The full description is available [here](https://docs.google.com/document/d/1DCx2Duijb0COESCuxwEEK1j0BPe2cTIJ-AjtJxt3290/edit#heading=h.2hp5zbf0n3o3)
and should be considered when filling out this form.

Please, provide information on your compability status for each mandatory policy, and if possible also for recommended policies.
If you are not compatible, state what is lacking and what are your plans on how to achieve compliance.
For current xSDK member packages: If you were not compliant at some point, please describe the steps you undertook to fulfill the policy.This information will be helpful for future xSDK member packages.

**Website:**  https://github.com/spiral-software/spiral-software

### Mandatory Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**M1.** Support xSDK community GNU Autoconf or CMake options. |Full| SPIRAL uses CMake.|
|**M2.** Provide a comprehensive test suite for correctness of installation verification. |Full| SPIRAL has several test scripts to exercise many features. |
|**M3.** Employ userprovided MPI communicator (no MPI_COMM_WORLD). |Not Applicable| SPIRAL does not use MPI; SPIRAL does generate code to leverage MPI (if available). |
|**M4.** Give best effort at portability to key architectures (standard Linux distributions, GNU, Clang, vendor compilers, and target machines at ALCF, NERSC, OLCF). |Full| Yes, also supported on Windows, Linux, and MacOS.|
|**M5.** Provide a documented, reliable way to contact the development team. |Full| SPIRAL developers can be contacted via: [https://www.spiralgen.com/contact](https://www.spiralgen.com/contact).|
|**M6.** Respect system resources and settings made by other previously called packages (e.g. signal handling). |Full| Yes.  |
|**M7.** Come with an open source (BSD style) license. |Full| Use 2-clause BSD license. |
|**M8.** Provide a runtime API to return the current version number of the software. |Full| SPIRAL has a builtin in command [Version();] to report this information. |
|**M9.** Use a limited and well-defined symbol, macro, library, and include file name space. |Full| None.  |
|**M10.** Provide an xSDK team accessible repository (not necessarily publicly available). |Full| [https://github.com/spiral-software/spiral-software](https://github.com/spiral-software/spiral-software) |
|**M11.** Have no hardwired print or IO statements that cannot be turned off. |Full| None. |
|**M12.** For external dependencies, allow installing, building, and linking against an outside copy of external software. |Full| SPIRAL's profiler depends on **python** at runtime; other than that a standard C compiler and **cmake** are required to install/build.  SPIRAL does not contain any other package's source code within  |
|**M13.** Install headers and libraries under \<prefix\>/include and \<prefix\>/lib. |Full| SPIRAL does not need to install headers or libraries. |
|**M14.** Be buildable using 64 bit pointers. 32 bit is optional. |Full| Supports both 32 and 64 bit under same API. |
|**M15.** All xSDK compatibility changes should be sustainable. |Full| All xSDK compatibility changes are part of the main DTK repository. |
|**M16.** The package must support production-quality installation compatible with the xSDK install tool and xSDK metapackage. |No| SPIRAL configure and install support from Spack will be added in future. |

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** Have a public repository. |Full| [https://github.com/spiral-software/spiral-software](https://github.com/spiral-software/spiral-software) |
|**R2.** Possible to run test suite under valgrind in order to test for memory corruption issues. |Partial| valgrind run in development but tests not currently builtin. |
|**R3.** Adopt and document consistent system for error conditions/exceptions. |None| None. |
|**R4.** Free all system resources acquired as soon as they are no longer needed. |Full| Yes. |
|**R5.** Provide a mechanism to export ordered list of library dependencies. |No| This can be added in future. |

