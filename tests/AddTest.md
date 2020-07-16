How to Add Test(s) to SPIRAL
============================

SPIRAL has a suite of tests that can be run to validate current functioning of
the installed software.

The tests are contained in a **tests** folder off the SPIRAL root directory.
Beneath **tests** are several subfolders according to the category/type of
tests.  Tests may be added to any of these subfolders, or new folders created
for new category tests as descibed below.

### Test Hierarchy

The current test hierarchy may be viewed as follows:

```
    SPIRAL Root Direcory
        |
        +----tests
                |
                +----Basic
                |
                +----Advanced
                |
                +----OpenMp
                |
                +----Scalar-Transforms
                |
                +----X86-SIMD
```

The subfolders under **tests** correspond to the categories of tests.  New
test(s) may be added to an existing category by placing the test file in the
appropraite subfolder.  A new category may be added by creating a new subfolder
in the **tests** directory.

If a new subfolder is added, then add the subfolder to *tests/CMakeLists.txt*;
ensure appropriate checks are placed for tests or test categories that depend on
specific platform features or libraries (the existing examples may provide
guidance).

Individual tests added to any subfolder simply need to be added to the
**TESTS_ALL_TESTS** set in the *CMakeLists.txt* file in the subfolder.

### Running Tests

Tests are run using **ctest** (if **make** is available **ctest** can be run by invoking *make test*).  **ctest** has options to control which tests are to be run (see **ctest** man page for full details).  A few simple ones are presented here:

| Option | Description |
|----:|:---------|
| -L | Run tests that match the *label* following the option
| -R | Run tests that match the regular expression following the option
| -E | Exclude tests that match the regular expression following the option
| -LE | Run all tests **except** those that match the *label* following the option

NOTE: All tests are automatically assigned a *label* property that matches the
subfolder in which they are defined.

For example:
```
ctest -L Basic                     # Run tests with label "Basic"
ctest -R FFT                       # Run tests with FFT in the name
ctest -R '.*k.e.*'                 # Run tests with matching arbitrary RE (quote the RE)
```


