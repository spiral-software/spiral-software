#!/usr/bin/env python3

# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

import argparse
import os
import glob
import sys
import tempfile
import shutil
import platform
import subprocess

MinPythonVersion    = (3, 6)
ProfilerName        = 'spiralprofiler'
ProfilerVersion     = '1.0.0'

def slurmAvailable():
    """Return True if Slurm commands are present and usable."""
    return shutil.which("sbatch") is not None and shutil.which("squeue") is not None

def filesToSend():
    pattern = os.path.join(srcdir, '*.[ch]')
    return glob.glob(pattern)

def cleanup():
    # delete temp directory; don't delete a named/defined builddir
    if not keeptemp and not builddir:
        os.chdir(workdir)
        shutil.rmtree(tempworkdir, ignore_errors=True)

import time

def runWithSlurm(command, tempworkdir, target, account=None, partition=None, walltime="00:02:00"):
    """Submit a Slurm job to run the given executable from tempworkdir."""
    batch_script = os.path.join(tempworkdir, "job.slurm")
    with open(batch_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=spiralrun\n")
        if account:
            f.write(f"#SBATCH --account={account}\n")
        if partition:
            f.write(f"#SBATCH --partition={partition}\n")
        f.write(f"#SBATCH --time={walltime}\n")
        f.write("#SBATCH --cpus-per-task=1\n")
        if "cuda" in target.lower():
            f.write("#SBATCH --partition=GPU-shared\n")
            f.write("#SBATCH --gres=gpu:1\n")
            ##  Accept default memory for now
            ##  f.write("#SBATCH --mem=16G\n")
            f.write("#SBATCH --ntasks=4\n")
        else:
            f.write("#SBATCH --partition=RM-shared\n")
            ##  Accept default memory for now
            ##  f.write("#SBATCH --mem=1G\n")
            f.write("#SBATCH --ntasks=1\n")
            
        f.write("#SBATCH --output=slurm.out\n")
        f.write("#SBATCH --error=slurm.err\n")
        f.write(f"{command}\n")

    sub = subprocess.run(["sbatch", batch_script], capture_output=True, text=True)
    if sub.returncode != 0:
        print("Error submitting job:", sub.stderr)
        return sub.returncode

    # Extract job ID
    jobId = None
    for token in sub.stdout.split():
        if token.isdigit():
            jobId = token
            break

    if not jobId:
        print("Could not parse job ID from sbatch output:", sub.stdout)
        return 1

    # Wait for job to finish
    while True:
        check = subprocess.run(["squeue", "-j", jobId], capture_output=True, text=True)
        if len(check.stdout.strip().splitlines()) <= 1:
            break
        time.sleep(2)

    return 0


if sys.version_info < MinPythonVersion:
    sys.exit('Error: Python %s.%s or later is required.\n' % MinPythonVersion)


bindir  = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(bindir)
def_request = 'time'
def_srcdir  = basedir
def_prefix  = None
def_workdir = basedir

if sys.platform == 'win32':
    def_target  = 'win-x86-vcc'
elif sys.platform == 'linux':
        if os.uname().machine == 'ppc64le':
                def_target = 'linux-ppc64le-gcc'
        elif os.uname().machine.startswith('armv7'):
                def_target = 'linux-arm-gcc'
        else:
                def_target  = 'linux-x86'
elif sys.platform == 'darwin':
    def_target  = 'darwin-x86'
else:
    def_target = None

parser = argparse.ArgumentParser(prog=ProfilerName)

parser.add_argument('-r', '--request', dest='request', default=def_request, help="Type of request (default: {0})".format(def_request))
parser.add_argument('-t', '--target',  dest='target',  default=def_target,  help="Target architecture (default: {0})".format(def_target))
parser.add_argument('-d', '--srcdir', dest='srcdir', default=def_srcdir, help='Directory containing generated source code.')
parser.add_argument('-D', '--debug', dest='debug', help='Print debug messages.', action='store_true')
parser.add_argument('-f', '--forward',  dest='remote', help='Forward to remote target.')
parser.add_argument('-k', '--keeptemp', dest='keeptemp', help='Keep temporary directories.', action='store_true')
parser.add_argument('-P', '--prefix',  dest='prefix', default=def_prefix, help='Temporary directory name prefix.')
parser.add_argument('-w', '--workdir', dest='workdir', default=def_workdir, help='Working directory subtree root, contains targets and tempdirs.')
parser.add_argument('-v', '--version', dest='version', help='Show version info', action='store_true')
parser.add_argument('-a', '--account', dest='account', help='HPC Account name, for submitting Slurm jobs.')
parser.add_argument('-p', '--partition', dest='partition', help='HPC Partition name, for submitting Slurm jobs.')
parser.add_argument('-b', '--builddir', dest='builddir',
                    help='Optional persistent build directory. If specified, it will be reused and cleared before each run.')

cmdarglist = sys.argv[1:]
extraargs = os.getenv('PROFILER_LOCAL_ARGS')
if isinstance(extraargs, str):
    cmdarglist.extend(extraargs.split())

args = vars(parser.parse_args(cmdarglist))

# -v option: print version info and exit
if args.get('version', False):
    print('Spiral Profiler', ProfilerVersion)
    print('  ', sys.argv[0])
    print('  ', platform.python_implementation(), platform.python_version())
    sys.exit(0)

debug    = args.get('debug', False)
request  = args.get('request', def_request)
target   = args.get('target', def_target)
srcdir   = os.path.realpath(args.get('srcdir', def_srcdir))
keeptemp = args.get('keeptemp', False)
prefix   = args.get('prefix', def_prefix)
workdir  = os.path.realpath(args.get('workdir', def_workdir))
remote   = args.get('remote', None)
account  = args.get('account', None)
partition = args.get('partition', None)
builddir = args.get('builddir', None)

if debug:
    print("Options:")
    print("  Profiler root:", basedir)
    print("  request:", request)
    print("  target:", target)
    print("  srcdir:", srcdir)
    print("  keeptemp:", keeptemp)
    print("  prefix:", prefix)
    print("  workdir:", workdir)
    if remote:
        print("  remote:", remote)
    if account:
        print("  account:", account)
    if partition:
        print("  partition:", partition)
    if builddir:
        print("  Named build directory:", builddir)

# verify source directory exists
if not os.path.exists(srcdir):
    sys.exit('Error: Cannot find source directory: "' + srcdir + '"')

# look for specified target
targetbasedir = os.path.join(workdir, 'targets')
if not os.path.exists(targetbasedir):
    sys.exit('Error: Cannot find directory: "' + targetbasedir + '"')
    
if remote:
    targetbasedir = os.path.join(targetbasedir, 'remote')
    if not os.path.exists(targetbasedir):
        sys.exit('Error: Cannot find directory: "' + targetbasedir + '"')
    targetdir = os.path.join(targetbasedir, remote)
else:
    targetdir = os.path.join(targetbasedir, target)
    
if not os.path.exists(targetdir):
    sys.exit('Error: Cannot find target directory: "' + targetdir + '"')

reqbase = 'forward' if remote else request

# make sure target supports request
cmdfile = os.path.join(targetdir, reqbase + '.cmd')
if not os.path.exists(cmdfile):
    sys.exit('Error: Cannot find request file: "' + cmdfile + '"')

# find and possibly create the subdirectory of temp dirs
tempdirs = os.path.join(workdir, 'tempdirs')
os.makedirs(tempdirs, mode=0o777, exist_ok=True)

# create temporary work directory; use builddir if provided
if builddir:
    tempworkdir = os.path.realpath(os.path.expanduser(builddir))
    os.makedirs(tempworkdir, mode=0o777, exist_ok=True)

    ##  Clear any contents of builddir before reuse
    for fname in os.listdir(tempworkdir):
        fpath = os.path.join(tempworkdir, fname)
        try:
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
        except Exception as e:
            print(f"Warning: Could not remove {fpath}: {e}")
else:
    tempworkdir = tempfile.mkdtemp(None, prefix, tempdirs)

# copy files from source directory to temporary work directory
filelist = filesToSend()
if debug:
    print("source files:", filelist)
    
for fname in filelist:
    try:
        shutil.copy(fname, tempworkdir)
    except:
        cleanup()
        sys.exit('Error: Could not copy ' + fname + ' to ' + tempworkdir)

##  Copy the targets/common/CMakeLists.txt to the temporary work directory
cmakeCommon = os.path.join(basedir, 'targets', 'common', 'CMakeLists.txt')
try:
    shutil.copy(cmakeCommon, tempworkdir)
except Exception as e:
    print("Error copying CMakeLists.txt:", str(e))
    cleanup()
    sys.exit(1)
    
command = cmdfile
if remote:
    command = command + ' -r ' + request + ' -t ' + target

# run the request
if debug:
    print("temporary work directory:", tempworkdir);
    print("command:", command);

os.chdir(tempworkdir)

##  Build phase first; execution may be directly running script or submitting Slurm job

try:
    buildCmd = command + ' build'
    env = os.environ.copy()
    env["SPIRAL_PROFILER_ROOT"] = basedir
    subret = subprocess.run ( buildCmd, shell=True, capture_output=True, text=True, env=env)
    res = subret.returncode

    if debug:
        print("stdout:", subret.stdout)
        print("stderr:", subret.stderr)

except Exception as e:
    print("Exception:", str(e))
    cleanup()
    sys.exit('Error: Could not subprocess.run(buildCmd)')

if (res != 0):
    cleanup()
    sys.exit(res)

##  Run phase

try:
    if slurmAvailable():
        print("Slurm detected -- submitting job to scheduler.")
        res = runWithSlurm(command, tempworkdir, target,
                           account=account,      # or None
                           partition=partition,  # or None
                           walltime="00:05:00")
    else:
        subret = subprocess.run ( command, shell=True, capture_output=(sys.platform == 'win32') )
        res = subret.returncode
except:
    cleanup()
    sys.exit(1)

if res != 0:
    cleanup()
    sys.exit(res)

# copy results back to source directory
resfile = request + '.txt'
try:
    shutil.copy(resfile, srcdir)
except:
    cleanup()
    sys.exit(f'Error: Could not copy results {resfile} to {srcdir}')

cleanup()

sys.exit(res)

