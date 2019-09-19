SPIRAL
======

This is the source tree for SPIRAL.  It builds and runs on Windows, Linux, and macOS.

## Building SPIRAL
### Prerequisites

#### C Compiler and Build Tools

SPIRAL builds on Linux/Unix with **gcc** and **make**, on Windows it builds with **Visual Studio**.

For macOS SPIRAL requires version 10.14 (Mojave) or later of macOS, with a compatible version of **Xcode** and
and **Xcode Command Line Tools**. 

#### CMake 3

Use the most recent version of CMake 3 available for your platform.  You can download CMake from [cmake.org](http://cmake.org/download/).

#### Python 3

The SPIRAL Profiler requires **Python 3**, which you can get from [python.org](http://python.org/downloads/).

### Building on Linux, macOS, and Other Unix-Like Systems

From the top directory of the SPIRAL source tree:
```
mkdir build
cd build
cmake ..
make install
```

Use the **spiral** script in the top directory to start SPIRAL.  You can run it from that directory or set your path to include
it and run **spiral** from elsewhere.  The actual executable is ```gap/bin/gap```, but it must be started
with the **spiral** script in order to intialize and run correctly.

NOTE: The **spiral** script is automatically created, with deaults appropriate to your environment, during the build process, at the *install* step.

#### Debug Version on Linux/Unix

To build a debug version of SPIRAL for use with **gdb**, from the top directory:
```
mkdir buildd
cd buildd
cmake -DCMAKE_BUILD_TYPE=Debug ..
make install
```

This will put the executable, **gapd**, in ```gap/bin```.  Use the **spirald** script to run SPIRAL in **gdb**.

### Building on Windows

In the top directory of the SPIRAL source tree, make a directory called **build**.  From a terminal window in the **build**
directory enter one of the following commands, depending on your version of Visual Studio.  See the 
[CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#visual-studio-generators)
if your version isn't shown here.

```
cmake -G "Visual Studio 14 2015" -A x64 ..

cmake -G "Visual Studio 15 2017" -A x64 ..

cmake -G "Visual Studio 16 2019" -A x64 ..
```

When CMake is finished, open the new **SPIRAL.sln** in the **build** directory with Visual Studio.  Select the Release or Debug configuration,
then right click on **INSTALL** in the Solution Explorer window and select **Build** from the popup menu.

Use **spiral.bat** to launch Spiral.  You can create a shortcut to 
it (right click -> Create shortcut) and move the shortcut to a convenient location, like the Desktop, 
renaming it if you desire.  Then you can edit the shortcut's properties (right click -> Properties) and 
set **Start in** to some directory other than the repository root.  The **Start in** directory is the 
default location for SPIRAL to write generated files.  You can also add the top directory of the SPIRAL source tree
to your path and run the batch script as **spiral** from a command window or script.

To debug SPIRAL on Windows, build and install the Debug version, use **spiral_debug.bat** to start SPIRAL, then in Visual Studio use
**Debug->Attach to Process...** to connect the debugger to **gapd.exe**.

## Testing SPIRAL

SPIRAL is released with a suite of self-tests.  The tests are automatically
configured and made available when **CMake** configures the installation.
Normally, all tests are configured; however, if a platform does not provide
required support for a tests it will not be configured (e.g., X86/SIMD tests
are not configured for ARM processor).

### Running Tests on Linux/Unix

All tests can be run using **make test** from the `<build>` folder, as follows:
```
cd build
make test
```

The **CMake** test driver program, **ctest**, can be used to exercise control
over which tests are run.  You can run a specific named test, tests matching a
specific type, or all tests using **ctest**.  A couple of examples are shown
here (refer to the CMake/ctest documetation for a full explanation of all
options).  The ` -L ` and ` -R ` options take a regular expression argument to
identify which tests to run.  ` -L ` matches against a label associated to
each test, while ` -R ` matches against the name of the test.

```
cd build
ctest -R Simple-FFT             # Run the test matching the name "Simple-FFT"
ctest -R [.]*FFT[.]*            # Run all test(s) with "FFT" in the name
ctest -L Basic                  # Run all test(s) with label "Basic"
ctest                           # Run all tests
```

### Disabling Tests

It is possible to disable the setup of tests when configuring the installation
with **CMake**.  There are five categories of tests: Basic, Advanced, Scalar-Transforms, OpenMP, and
X86-SIMD.  Each category of test may be turned off by adding a define on the
cmake command line, as follows:
```
-DTESTS_RUN_BASIC=OFF                       # Turn off basic tests
-DTESTS_RUN_ADVANCED=OFF                    # Turn off Advanced tests
-DTESTS_RUN_SCALAR_TRANSFORMS=OFF           # Turn off FFT tests
-DTESTS_RUN_OPENMP=OFF                      # Turn off Search tests
-DTEST_RUN_X86_SIMD=OFF                     # Turn off X86-SIMD tests
```

### Running Tests on Windows

**make** is genarally not available on Windows, so we need to use **ctest** in
order to run the tests.  For Windows **ctest** needs to be informed which
confiuration was built, using the ` -C <config> ` option; where ` <config> `
is typically ` Release ` or ` Debug `. Other than this caveat, tests are configured, run, or
inhibited exactly the same on Windows as Linux/Unix.  For example:
```
cd build
ctest -C Release -R Simple-FFT     # Run the test matching the name "Simple-FFT" on Release build
ctest -C Release -R [.]*FFT[.]*    # Run all test(s) with "FFT" in the name on Release build
ctest -C Debug -L Basic            # Run all test(s) with label "Basic" on Debug build
ctest -C Release                   # Run all tests on Release build
```

### Debugging Failed Tests

Should a test fail you can view the complete test inputs/outputs generated
during the test run.  Beneath the `<build>` folder is a **Testing/Temporary**
folder where **ctest** logs all the generated information.  The data for the
last test run is contained in the file **LastTest.log**.

Spiral Profiler
--------------

The performance measurement functionality is in the **profiler** subdirectory.  It has the 
mechanisms for asynchronously measuring code performance.  Refer to the **README** in 
the **profiler** directory.

The interface from SPIRAL to the profiler is in **namespaces/spiral/profiler**.
