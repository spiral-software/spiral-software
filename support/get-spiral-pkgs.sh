#! /bin/bash

##  Get spiral and its required packages {fftx, simt, jit, and mpi} and FFTX
##  Assumptions:
##      SPIRAL_HOME is set -- exit if not
##      if the directory exists do a git pull (develop branch) in that directory
##      If it doesn't exist then clone spiral-software (develop branch) into that location (assume SPIRAL_HOME/.. exists)
##      For each of the three aforementioned packages:
##          Clone into a folder beneath namespaces/packages (or do a git pull to refresh)
##      If FFTX_HOME is set, get (or clone) FFTX into that location
##          
##      This script is only intended to get 'clean' software from the main spiral-software repos.
##      In particular, it won't do the right thing for forks or when changes exist in the local tree
##      (see spiral-git-update.sh if you need to update fork)

get_from_github() {
    local dir=$1
    local pkg=$2
    local branch=$3
    
    if [ -d "$dir" ]; then
        ##  Directory exists
        pushd $dir
        if [ -z "$branch" ]; then
            branch=`git remote show origin | grep 'HEAD branch' | awk '{print $NF}'`
        fi
        git checkout $branch
        git pull
    else
        ##  directory does not exist
        pushd `dirname $dir`
        git clone https://github.com/spiral-software/$pkg `basename $dir`
        cd $dir
        if [ -z "$branch" ]; then
            branch=`git remote show origin | grep 'HEAD branch' | awk '{print $NF}'`
        fi
        git checkout $branch
        git pull
    fi
    popd
}

##  get Spiral and packages, warn and exit with error if SPIRAL_HOME not set
##  get FFTX if FFTX_HOME is defined; otherwise remind user to get it manually later
##  Accept an a command line argument naming the branch to use for the repos (none = default { main, master } )

if [ $# -gt 0 ]; then
    ##  A command line argument was given
    branch=$1
else
    branch=""
fi

if [ -n "$SPIRAL_HOME" ]; then
    ##  SPIRAL_HOME is defined
    get_from_github $SPIRAL_HOME "spiral-software" $branch
else
    ##  SPIRAL_HOME is not defined -- print error & exit
    echo "Please set SPIRAL_HOME environment variable to the location you want Spiral to be located, exiting"
    exit 1
fi

pushd $SPIRAL_HOME/namespaces/packages

for xx in fftx simt jit mpi; do
    get_from_github $xx spiral-package-$xx $branch
done

popd

if [ -n "$FFTX_HOME" ]; then
    ##  FFTX_HOME is defined
    get_from_github $FFTX_HOME "fftx" $branch
else
    ##  FFTX_HOME is not defined -- tell user to get it manually
    echo "FFTX_HOME environment variable is not set; get fftx manually yourself"
    exit 0
fi

exit 0
