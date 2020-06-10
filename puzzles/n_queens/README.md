# N-Queens - 92 Solutions In Prolog¶
- https://www.kaggle.com/jamesmcguigan/n-queens-92-solutions-in-prolog

This code solves the N-Queens problem in Prolog using the CLP(FD): Constraint Logic Programming over Finite Domain Library

- `nqueens()` creates a 2D array datastructure, representing the board coordinates of each queen
- `applyConstraints()` recursively iterates through each queen on the board
- `checkConstraints()` applies the constraints: no two queens on same row/column/diagonal; and recurses through the list of remaining queens
- `optimizeQueens()` hardcodes each queen to live in a named row, this greatly reduces the computational complexity of the problem
    - Note: it is not possible to pass Index + 1 into a prolog function, instead it must be declared and solved first as its own variable: NextIndex is Index + 1
- `print_board()` + `print_line()` render the ASCII graphics in a functional manner
- `all_nqueens()` uses `findall()` to solve `nqueens()` whilst repeatedly adding previous solutions as future constraints
- print_nqueens_all(N) is the main display/execution function

## Install and Execute
```
brew install swi-prolog  # swipl
brew install gnu-prolog  # gprolog

swipl -f nqueens.prolog -t 'print_nqueens_all(8)' > nqueens.txt
```


