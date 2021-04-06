Contributing to a SPIRAL Package
================================

This guide provides instructions on how to collaborate to add a new feature or enhancement
or provide a bug fix to a SPIRAL package.

The basic model is to fork [or clone] the public repository for SPIRAL and then fork the
SPIRAL package you wish to change, add your changes, build test(s) showing the deficieny
or bug and also showing how your additions work to remedy that item, then create a pull
request (PR) to the SPIRAL team to review and incorporate your changes.

NOTE:  If you intend to make changes to SPIRAL you must fork it, please refer to the
instructions for [Contributing to SPIRAL](https://spiral-software.github.io/spiral-software/contribute.html)

### SPIRAL Repository

The public version of the SPIRAL software repository is available
[**here**](https://github.com/spiral-software/spiral-software.git).  Fork or clone this
repository then go to directory **spiral-software/namespaces/packages** where you'll
install the SPIRAL package.

### Fork the Package Repository

SPIRAL packages are intended to be installed into the
**spiral-software/namespaces/packages** folder of SPIRAL; a package must be installed at
the suffix portion of the full package name, e.g., **spiral-package-fftx** must be installed
as **fftx**.  For the reminder of this guide we'll use **fftx** as the example; however, the
instructions are aplicable to all packages.

1.  On GitHub navigate to the package repository [SPIRAL package:
spiral-package-fftx](https://github.com/spiral-software/spiral-package-fftx.git). 
2.  In the top-right corner of the page click the **Fork** button.  This will create a
private repository for your use, called:
``github.com/<your_username>/spiral-package-fftx.git``, where **\<your_username\>** is
your GitHub username. 
3.  Clone the new repository to your machine so you can develop, into the packages folder,
e.g., use **git** as follows: ``git clone
github.com/<your_username>/spiral-package-fftx.git fftx`` 
4.  It is good practice to regularly sync your fork with the upstream repository.  Using the command line:
```
   cd fftx
   git remote -v		## this will show:
      > origin   https://github.com:/<your_username>/spiral-package-fftx.git (fetch)
      > origin   https://github.com:/<your_username>/spiral-package-fftx.git (push)
      
   git remote add upstream https://github.com/spiral-software/spiral-package-fftx.git	## Sync with upstream
   git remote -v		## verify added
      > origin   https://github.com:/<your_username>/spiral-package-fftx.git (fetch)
      > origin   https://github.com:/<your_username>/spiral-package-fftx.git (push)
      > upstream	https://github.com/spiral-software/spiral-package-fftx.git (fetch)
      > upstream	https://github.com/spiral-software/spiral-package-fftx.git (push)
```

Now, you can keep your fork synced with the upstream repository, using **git**:
```
git fetch upstream                    ## pulls any updates from upstream to local branch called upstream/master
git branch <your_branch>              ## create a unique branch for your changes
git checkout <your_branch>            ## switch to the your branch 
git merge upstream/master             ## merges any changes from the upstream to the local
```

### Develop / Make Changes

When making changes or adding bug fixes please limit each set of changes to **one**
feature or **one** bug fix.  Doing so will greatly ease the task of reviewing and
incorporating your changes into the main product.  You *must* include with your changes
test(s) that demonstrate the correct functioning of the feature or bug fix (in the case of
bug fixes it'll also help to have a test or script demonstrating the bug in the original
version).

Please create a unique branch for your changes:  It'll help if the branch is named
something descriptive, e.g., **bug_memory_leak**.  Using a named branch leaves the master
pure as you develop.  Before submitting your changes to the SPIRAL team you should run
*and pass all* the tests available with SPIRAL and the package you are working with.
Changes that break existing features or that cannot pass the existing test suite **will be
rejected**. 

When your changes are complete, add changed or new files and commit the changes in the
normal manner.  When you are ready to push  <your_branch> for the first time you need to
let **git** know about the upstream branch, e.g., use this command:<br>
``git push --set-upstream origin <your_branch>``

### Create the Pull Request (PR)

First, what is a PR?  Pull requests let you tell others about changes you've pushed to a branch in a repository on GitHub.  A separate pull request (on a unique branch) should be made for each completed **feature** or **bug fix**.  Once a pull request is opened, you can discuss and review the potential changes with collaborators and add follow-up commits before your changes are merged into the base branch.

The important thing to note here is one completed **feature** per pull request.  The changes one proposes in a PR should correspond to one feature/bug fix/extension/etc.  One can create PRs with changes relevant to different ideas, however reviewing such PRs becomes tedious and error prone.  If possible, please follow the one-PR-one-package/feature rule.

To create the pull request, do the following:

1.  Navigate to the repository on GitHub (i.e., your fork of the original).
2.  In the **Branch** menu choose the branch that contains your committed changes.
3.  Above the list of files, click **Pull request**.
4.  The pull request shows dropdowns for branch selection: The **base repository** should
be the master repository from which your clone was made.  The **base** branch dropdown
selects the branch to which you want your changes added (normally **main**).  The **head
repository** is your clone of the original.  In the **compare** branch dropdown select the
branch to which you committed your changes (i.e., *\<your_branch\>*). 
5.  Enter a title and brief description for your pull request.
6.  To create a pull request that is ready for review, click the **Create Pull Request** button.

After submitting the pull request it is available for review by the SPIRAL team.  In the course of the review comments or issues may be raised that require resolution before the changes can be committed to the main branch.  GitHub will notify the submitter of any such comments.
