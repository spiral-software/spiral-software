SPIRAL
======

This is the source tree for SPIRAL.  It builds and runs on both Windows and Linux.

## Building SPIRAL
### Prerequisites

#### C Compiler and Build Tools

SPIRAL builds on Linux/Unix with **gcc** and **make**, on Windows it builds with **Visual Studio**.

#### CMake 3

Use the most recent version of CMake 3 available for your platform.  You can download CMake from [cmake.org](http://cmake.org/download/).

### Building on Linux and Other Unix-Like Systems

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

Spiral Profiler
--------------

The performance measurement functionality is in the **profiler** subdirectory.  It has the 
mechanisms for asynchronously measuring code performance.  Refer to the **README** in 
the **profiler** directory.

The interface from SPIRAL to the profiler is in **namespaces/spiral/profiler**.
