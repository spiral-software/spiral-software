Contributing to SPIRAL
======================

This guide provides instructions on how to collaborate to add a new feature or enhancement or provide a bug fix to SPIRAL.

The basic model is to fork the public repository for SPIRAL, add your changes, build test(s) showing the deficieny or bug and also showing how your additions work to remedy that item, then create a pull request (PR) to the SPIRAL team to review and incorporate your changes.

## SPIRAL Repository

The public version of the SPIRAL software repository is available [**here**](https://github.com/spiral-software/spiral-software.git).

### Fork the Repository

1.  On GitHub navigate to the [SPIRAL repository](https://github.com/spiral-software/spiral-software.git).
2.  In the top-right corner of the page click the **Fork** button.  This will create a private repository for your use, called: <br>*github.com/\<your_username\>/spiral-software.git*, where **\<your_username\>** is your GitHub username.
3.  Clone the new repository to your machine so you can develop, e.g., use **git** as follows: <br>*git clone github.com/\<your_username\>/spiral-software.git*
4.  It is good practice to regularly sync your fork with the upstream repository.  Using the command line:
```
   cd <folder_you_created_when_cloning_above>
   git remote -v		## this will show:
      > origin   https://github.com:/<your_username>/spiral-software.git (fetch)
      > origin   https://github.com:/<your_username>/spiral-software.git (push)
      
   git remote add upstream https://github.com/spiral-software/spiral-software.git	## Sync with upstream
   git remote -v		## verify added
      > origin   https://github.com:/<your_username>/spiral-software.git (fetch)
      > origin   https://github.com:/<your_username>/spiral-software.git (push)
      > upstream	https://github.com/spiral-software/spiral-software.git (fetch)
      > upstream	https://github.com/spiral-software/spiral-software.git (push)
```

Now, you can keep your fork synced with the upstream repository, using **git**:
```
git fetch upstream                    ## pulls any updates from upstream to local branch called upstream/master
git branch <your_branch>              ## create a unique branch for your changes
git checkout <your_branch>            ## switch to the your branch 
git merge upstream/master             ## merges any changes from the upstream to the local
```

### Develop / Make Changes

When making changes or adding bug fixes please limit each set of changes to **one** feature or **one** bug fix.  Doing so will greatly ease the task of reviewing and incorporating your changes into the main product.  You *must* include with your changes test(s) that demonstrate the correct functioning of the feature or bug fix (in the case of bug fixes it'll also help to have a test or script demonstrating the bug in the original version).

Please create a unique branch for your changes:  It'll help if the branch is named something descriptive, e.g., **bug_memory_leak**.  Using a named branch leaves the master pure as you develop.  Before submitting your changes to the SPIRAL team you should run *and pass all* the tests available with SPIRAL.  Changes that break existing features or that cannot pass the existing test suite **will be rejected**.

When your changes are complete, add changed or new files and commit the changes in the normal manner.  When you are ready to push <your_branch> for the first time you need to let **git** know about the upstream branch: use this command:
```
git push --set-upstream origin <your_branch>
```

### Create the Pull Request (PR)

First, what is a PR?  Pull requests let you tell others about changes you've pushed to a branch in a repository on GitHub.  A separate pull request (on a unique branch) should be made for each completed **feature** or **bug fix**.  Once a pull request is opened, you can discuss and review the potential changes with collaborators and add follow-up commits before your changes are merged into the base branch.

The important thing to note here is one completed **feature** per pull request.  The changes one proposes in a PR should correspond to one feature/bug fix/extension/etc.  One can create PRs with changes relevant to different ideas, however reviewing such PRs becomes tedious and error prone.  If possible, please follow the one-PR-one-package/feature rule.

To create the pull request, do the following:

1.  Navigate to the repository on GitHub (i.e., your fork of the original).
2.  In the **Branch** menu choose the branch that contains your committed changes.
3.  Above the list of files, click **Pull request**.
4.  The pull request shows dropdowns for branch selection: The **base repository** should be the master repository from which your clone was made.  The **base** branch dropdown selects the branch to which you want your changes added (normally **master**).  The **head repository** is your clone of the original.  In the **compare** branch dropdown select the branch to which you committed your changes (i.e., *\<your_branch\>*).
5.  Enter a title and brief description for your pull request.
6.  To create a pull request that is ready for review, click the **Create Pull Request** button.

After submitting the pull request it is available for review by the SPIRAL team.  In the course of the review comments or issues may be raised that require resolution before the changes can be committed to the master branch.  GitHub will notify the submitter of any such comments.
