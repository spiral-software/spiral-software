SPIRAL Profiler
===============

SPIRAL calls the profiler to time candidate kernels during a search.

The profiler requires Python 3 (>= 3.3).

On Windows you will need an instance of **Gnu Make**, which you can download from https://sourceforge.net/projects/gnuwin32/files/make/3.81/

Configuration
-------------

Open a terminal window in the profiler's **bin** directory and enter: ```spiralprofiler -h```

If things are working you should see the detailed list of command line options.  Check the default target architecture, which should correspond to your host environment.

### Working Directory

The "working directory" is the directory that contains the **targets** subdirectory and where the profiler creates the **tempdirs** subdirectory.  The default location is the same directory that contains **bin**.  If you want to move the working directory to somewhere else, copy the **targets** subdirectory, along with its contents, to the new location.

Use the **--workdir** command line option to specify a location other than the default.

### Temporary Directories

With each request the profiler creates a temporary directory under **tempdirs**  to build and run the test.  The default behavior is to delete the temporary directory once the results are returned to SPIRAL.  Use the **--keeptemp** command line option to leave the directories in place for later inspection.

### The PROFILER_LOCAL_ARGS Environment Variable

The profiler checks for an environment variable called **PROFILER_LOCAL_ARGS** for additional arguments to add to the command line.  Use it to specify any arguments you want to always add to the end of the argument list.

When first getting the profiler installed and configured, the **debug** and **keeptemp** options can be helpful.  For linux, add this to the **spiral** startup script in SPIRAL's root directory:
```
export PROFILER_LOCAL_ARGS="--debug --keeptemp"
```
On windows, add this to **spiral.bat** in SPIRAL's root directory:
```
set PROFILER_LOCAL_ARGS=--debug --keeptemp
```

How the Profiler Works
----------------------

Besides the working directory, there is some other key profiler terminology:

* ***source directory*** -- where SPIRAL places source code to test.  Specified with **--srcdir** argument.  The default is the working directory.

* ***target*** -- a directory containing an environment-specific Makefile, commands, and related files.  Specified with the **--target** argument.  The default depends on machine and OS.

* ***request*** -- the name for the requested action, such as **time** or **matrix**.  Specified with the **--request** argument.  The default is **time**. 

After a sanity check on parameters and its operating environment, the profiler:

1. creates a new temporary directory
1. copies code files from the source directory to the temporary directory
1. from within the temporary directory, calls **&lt;target>/&lt;request>.cmd**
1. copies **&lt;request>.txt** from the temporary directory to the source directory
1. cleans up and exits

The most common profiler request from SPIRAL is **time**, which runs several iterations of the compiled code and returns the average (mean) cycle count.  In the target directory is a **time.cmd** script, and the results are written to **time.txt**.  The **time** request expects two files in source directory:

* testcode.c
* testcode.h

On completion of a successful run, **time.txt**, containing the average cycle count, will be in the source directory.  SPIRAL reads the result from **time.txt**.

Simple Test
-----------

Start SPIRAL and enter the following script:
```
opts := SpiralDefaults;
t := DFT(32, -1);
rt := RandomRuleTree(t, opts);
s := SumsRuleTree(rt, opts);
c := CodeSums(s, opts);
CMeasure(c, opts);
```

You should see something like this:
```
spiral> CMeasure(c, opts);
spiralprofiler -d /tmp/2764 -r time -P 2764_
337.0
spiral> 
```
In this example **2764** is SPIRAL's PID.  SPIRAL put the test code in **/tmp/2764**, wants the profiler to **time** the code, wants the profiler to preface the temporary directory name with the PID, the resulting cycle count is 337.  If you have the **--debug** option set in **PROFILER_LOCAL_ARGS** there'll be some extra configuration info before the cycle count.

If you see something very different, read on...

Troubleshooting
---------------
The first thing to check is for obvious error messages in SPIRAL's output. On non-Windows systems, make sure that **time.cmd** in the target directory is executable.

Once you have a **testcode.c** and **testcode.h** from SPIRAL you can run the profiler from the command line to investigate problems.  Copy the two files from the source directory (the **-d** in SPIRAL's call to the profiler) into the profiler root directory.  From a command line run:
```
bin/spiralprofiler -D -k
```
Once you have a saved directory under **tempdirs** you can run the target request command outside of the profiler. Assuming your target is **linux-x86**, cd into the newest directory under **tempdirs** and enter:
```
touch testcode.c
../../targets/linux-x86/time.cmd
```
The target commands are meant to be run from within a temporary directory.

The target request commands and the associated Makefiles are based on generic setups.  Either or both may require minor edits to adjust to your configuration.  When making changes in the target directory you should run ```make clean``` before switching back to the temporary directory.




