# Kaggle ARC Abstraction and Reasoning Challenge

This codebase contains my entry and ongoing research into the Abstraction and Reasoning Corpus
- https://www.kaggle.com/c/abstraction-and-reasoning-challenge
- https://github.com/fchollet/ARC

This work was added to an ensemble of different approaches as part of  the _"Mathematicians + Experts"_ team
- https://www.kaggle.com/jesucristo
- https://www.kaggle.com/seshurajup
- https://www.kaggle.com/chihantsai
- https://www.kaggle.com/gamingconceited
- https://www.kaggle.com/jamesmcguigan

Public Leaderboard (22/914 = top 3% = Silver Medal)
- 0.941 (6/104) - https://www.kaggle.com/jamesmcguigan/arc-team-ensemble
- 0.990 (1/104) - https://www.kaggle.com/jamesmcguigan/arc-xgbsolver


# Install and Execute
```
./requirements.sh
source venv/bin/activate

python3 ./submission/kaggle_compile.py src/solver_multimodel/main.py                 | tee ./submission/submission.py
python3 ./submission/kaggle_compile.py src/ensemble/sample_sub/sample_sub_combine.py | tee ./submission/submission.py
```


# Team Ensemble
Solutions from other members of the _"Mathematicians + Experts"_ team
- [src/ensemble](src/ensemble)
- [team/ged](team/ged)
- [team/joe](team/joe)
- [team/seshu](team/seshu)


# Gallery
Jupyter Notebook visualization of the solved and unsolved results for each of the Solvers 
- [notebooks_gallery/solved](notebooks_gallery/solved)
- [notebooks_gallery/solved](notebooks_gallery/detects)


# Data Model
- Source: [src/datamodel/](src/datamodel/)

This is an object oriented data model around the dataset, 
allowing for static typing and ease of use navigating between
related datatypes. 


Conceptual Mapping:

- Competition: The collection of all Dataset in the competition
- Dataset:     An array of all Tasks in the competition
- Task:        The entire contents of a json file, outputs 1-3 lines of CSV
- ProblemSet:  An array of either test or training Problems
- Problem:     An input + output Grid pair
- Grid:        An individual grid represented as a numpy array
- CSV:         Export data model to `submission.csv` 


# Solver Abstract
- Source: [src/solver_abstract](src/solver_abstract)

Proof of concept: using `inspect.signiture()` figure out all possible permutations of `f(g(h(x)))`
and implement an IoC dependency injection solver.


# Solver MultiModel
- Original Notebook: https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers/
- Source: [src/solver_multimodel/](src/solver_multimodel/)

This is the main codebase.


## Core 
**Solver** implements an object oriented base to handle common code for looping, testing and generating 
solutions in the dataset, allowing subclasses to override lifecycle methods such as 
`detect()`, `fit()`, `solve_grid()`

**ProblemSetSolver** is a Solver subclass designed for algorithms requiring data access 
to the Task rather than just the current input Grid.

**ProblemSetEncoder** is a baseclass for autogenerating a feature map from a list of typed univariate functions

- [src/solver_multimodel/core/Solver.py](src/solver_multimodel/core/Solver.py)
- [src/solver_multimodel/core/ProblemSetEncoder.py](src/solver_multimodel/core/ProblemSetEncoder.py)
- [src/solver_multimodel/core/ProblemSetSolver.py](src/solver_multimodel/core/ProblemSetSolver.py)
- [src/solver_multimodel/core/XGBEncoder.py](src/solver_multimodel/core/XGBEncoder.py)


## Solvers

In order of complexity:

**DoNothingSolver** just returns the input grid
- [src/solver_multimodel/solvers/DoNothingSolver.py](src/solver_multimodel/solvers/DoNothingSolver.py)

**GlobSolver** indexes the training dataset and returns verbatim any problems seen in the training dataset 
- [src/solver_multimodel/solvers/GlobSolver.py](src/solver_multimodel/solvers/GlobSolver.py)

**ZoomSolver** applies `cv2.resize()` and `skimage.measure.block_reduce()` to problems whose input/output grid sizes are 
an integer multiple of each other
- [src/solver_multimodel/solvers/ZoomSolver.py](src/solver_multimodel/solvers/ZoomSolver.py)

**SingleColorSolver** + **BorderSolver** tests a list of functions to answer single color problems
- [src/solver_multimodel/solvers/SingleColorSolver.py](src/solver_multimodel/solvers/SingleColorSolver.py)
- [src/solver_multimodel/solvers/BorderSolver.py](src/solver_multimodel/solvers/BorderSolver.py)
- [notebooks_gallery/solved/SingleColorSolver.ipynb](notebooks_gallery/solved/SingleColorSolver.ipynb)

**GeometrySolver** performs a brute force search of numpy array functions
- [src/solver_multimodel/solvers/GeometrySolver.py](src/solver_multimodel/solvers/GeometrySolver.py)
- [notebooks_gallery/solved/GeometrySolver.ipynb](notebooks_gallery/solved/GeometrySolver.ipynb)

**TessellationSolver** applies nested geometry solutions to tessellation problems
- [src/solver_multimodel/solvers/TessellationSolver.py](src/solver_multimodel/solvers/TessellationSolver.py)
- [notebooks_gallery/solved/TessellationSolver.ipynb](notebooks_gallery/solved/TessellationSolver.ipynb)

**XGBGridSolver** generates a large multi-dimential featuremap to be solved by XGBoost. 
The featuremap includes each pixel's "view" of neighbouring pixels. This was able to autosolve a 
suprising number of problem cases, but also produces a large number of incorrect or close guesess that 
managed to test correctly against the train side the task.

Hyperopt Bayesian Hyperparameter Optimization was also performed on XGBoost.
- [src/solver_multimodel/solvers/XGBGridSolver.py](src/solver_multimodel/solvers/XGBGridSolver.py)
- [src/solver_multimodel/solvers/XGBGridSolver.hyperopt.py](src/solver_multimodel/solvers/XGBGridSolver.hyperopt.py)
- [notebooks_gallery/solved/XGBGridSolver.ipynb](notebooks_gallery/solved/XGBGridSolver.ipynb)

**XGBSingleColorSolver** solve simple problems using XGBoost in a subclassable way
- [src/solver_multimodel/solvers/XGBSingleColorEncoder.py](src/solver_multimodel/solvers/XGBSingleColorEncoder.py)
- [src/solver_multimodel/solvers/XGBSingleColorSolver.py](src/solver_multimodel/solvers/XGBSingleColorSolver.py)


# Utils
- Source: [src/utils/](src/utils/)

Various utility functions including `plot_task()` and `@np_cache()`


# Functions 
- Source: [src/functions/](src/functions/)

A range of different numpy.array queries and transformations


# Kaggle Compile
- Source: [submission/kaggle_compile.py](submission/kaggle_compile.py)

Kaggle Compile is a custom python concatenater that resolves local import statements and 
allows an IDE multi-file codebase to be compiled into a single-file Kaggle Kernel Script 