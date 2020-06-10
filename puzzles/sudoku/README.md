# Sudoku Solver

> Sudoku (数独, sūdoku, digit-single) (/suːˈdoʊkuː/, /-ˈdɒk-/, /sə-/, originally called Number Place) is a logic-based, combinatorial number-placement puzzle. The objective is to fill a 9×9 grid with digits so that each column, each row, and each of the nine 3×3 subgrids that compose the grid (also called "boxes", "blocks", or "regions") contain all of the digits from 1 to 9. The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a single solution.
>
> Completed games are always an example of a Latin square which include an additional constraint on the contents of individual regions. For example, the same single integer may not appear twice in the same row, column, or any of the nine 3×3 subregions of the 9×9 playing board.
>
> - https://en.wikipedia.org/wiki/Sudoku

## Simple Sudoku
```
 -  -  3 | -  2  - | 6  -  - 
 9  -  - | 3  -  5 | -  -  1 
 -  -  1 | 8  -  6 | 4  -  - 
---------|---------|---------
 -  -  8 | 1  -  2 | 9  -  - 
 7  -  - | -  -  - | -  -  8 
 -  -  6 | 7  -  8 | 2  -  - 
---------|---------|---------
 -  -  2 | 6  -  9 | 5  -  - 
 8  -  - | 2  -  3 | -  -  9 
 -  -  5 | -  1  - | 3  -  - 

 4  8  3 | 9  2  1 | 6  5  7 
 9  6  7 | 3  4  5 | 8  2  1 
 2  5  1 | 8  7  6 | 4  9  3 
---------|---------|---------
 5  4  8 | 1  3  2 | 9  7  6 
 7  2  9 | 5  6  4 | 1  3  8 
 1  3  6 | 7  9  8 | 2  4  5 
---------|---------|---------
 3  7  2 | 6  8  9 | 5  1  4 
 8  1  4 | 2  5  3 | 7  6  9 
 6  9  5 | 4  1  7 | 3  8  2 

solved in 0.22 s
```

## World's Hardest Sudoku 
- https://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html
```
 8  -  - | -  -  - | -  -  - 
 -  -  3 | 6  -  - | -  -  - 
 -  7  - | -  9  - | 2  -  - 
---------|---------|---------
 -  5  - | -  -  7 | -  -  - 
 -  -  - | -  4  5 | 7  -  - 
 -  -  - | 1  -  - | -  3  - 
---------|---------|---------
 -  -  1 | -  -  - | -  6  8 
 -  -  8 | 5  -  - | -  1  - 
 -  9  - | -  -  - | 4  -  - 

 8  1  2 | 7  5  3 | 6  4  9 
 9  4  3 | 6  8  2 | 1  7  5 
 6  7  5 | 4  9  1 | 2  8  3 
---------|---------|---------
 1  5  4 | 2  3  7 | 8  9  6 
 3  6  9 | 8  4  5 | 7  2  1 
 2  8  7 | 1  6  9 | 5  3  4 
---------|---------|---------
 5  2  1 | 9  7  4 | 3  6  8 
 4  3  8 | 5  2  6 | 9  1  7 
 7  9  6 | 3  1  8 | 4  5  2 

solved in 2.15 s
```
