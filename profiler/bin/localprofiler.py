#!/usr/bin/env python3

# Copyright (c) 2018-2021, Carnegie Mellon University
# See LICENSE for details

import argparse
import os
import sys
import tempfile
import shutil
import platform
import subprocess

MinPythonVersion    = (3, 6)
ProfilerName        = 'spiralprofiler'
ProfilerVersion     = '1.0.0'

def filesForRequest(request):
    if request == 'time':
        return ['testcode.c', 'testcode.h']
    if request == 'vector':
        return ['testcode.c', 'testcode.h']
    else:
        # TODO make this scalable/portable/configurable
        return ['testcode.c', 'testcode.h']
        
def cleanup():
    # delete temp directory
    if not keeptemp:
        os.chdir(workdir)
        shutil.rmtree(tempworkdir, ignore_errors=True)

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
parser.add_argument('-v', '--version', dest='version', help='Show version info',
action='store_true')

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

if debug:
    print("Options:")
    print("  request:", request)
    print("  target:", target)
    print("  srcdir:", srcdir)
    print("  keeptemp:", keeptemp)
    print("  prefix:", prefix)
    print("  workdir:", workdir)
    if remote:
        print("  remote:", remote)


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

# create temporary work directory
tempworkdir = tempfile.mkdtemp(None, prefix, tempdirs)

# copy files from source directory to temporary work directory
os.chdir(srcdir)
filelist = filesForRequest(request)
for fname in filelist:
    try:
        shutil.copy(fname, tempworkdir)
    except:
        cleanup()
        sys.exit('Error: Could not copy ' + fname + ' to ' + tempworkdir)

command = cmdfile
if remote:
    command = command + ' -r ' + request + ' -t ' + target

# run the request
if debug:
    print("temporary work directory:", tempworkdir);
    print("command:", command);

os.chdir(tempworkdir)
try:
    if sys.platform == 'win32':
        subret = subprocess.run(command, capture_output=True)
    else:
        subret = subprocess.run(command)
    res = subret.returncode
except:
    cleanup()
    sys.exit(1)

if (res != 0):
    cleanup()
    sys.exit(res)

# copy results back to source directory
resfile = request + '.txt'
try:
    shutil.copy(resfile, srcdir)
except:
    cleanup()
    sys.exit('Error: Could not copy results ' + resfile + ' to ' + srcdir)

cleanup()

sys.exit(res)

