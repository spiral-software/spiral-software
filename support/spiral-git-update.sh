#! /bin/bash

##  Update SPIRAL_HOME and spiral packages {fftx, simt, and mpi}
##  Assumptions:
##      spiral and the three aforementioned packages are forked from spiral-software
##      All are on the develop branch
##      This script will do a 'fetch upstream' and 'merge develop'
##      Finally, it'll push the change back to the local fork

update_from_remote() {
    local dir=$1
    pushd $dir
    foo=$(git remote -v | wc -l)
    if (( $foo > 2 )); then
        ##  fetch upstream and merge
        git fetch upstream
        git merge upstream/develop
        git push
        git pull
    else
        ##  no remote upstream, fetch/merge not possible
        echo "No remote upstream repository define, update not possible for:"
        git remote -v
    fi
    popd
}

##  Assume we are in the home directory ($SPIRAL_HOME)  -- could change to force
##  switch to $PSIRAL_HOME if desired.

update_from_remote "."
pushd namespaces/packages

for xx in fftx simt mpi jit; do
    if [ -d $xx ]; then
        update_from_remote $xx
    else
        echo "No package named $xx exists in `pwd`"
    fi
done

exit 0
