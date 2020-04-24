% Source: http://www.cse.scu.edu/~rdaniels/html/courses/Coen171/NQProlog.htm
queens([]).
queens([ Row/Col | Rest ]) :-
    queens(Rest),
    members(Col, [1,2,3,4,5,6,7,8]),
    safe( Row/Col, Rest ).

safe(_, []).  % Empty board is always safe
safe(Row/Col, [RowOther/ColOther | Rest]) :-
    Col =\= ColOther,                % not same column
    Row =\= RowOther,                % not same row
    Col1 - Col =\= RowOther - Row,   % check diagonals
    Col1 - Col =\= Row - RowOther,
    safe(Row/Col, Rest).             % recurse rest of list

member(X, [X | Tail]).
member(X, [Head | Tail]) :- member(X,Tail).

board() :- [ 1/C1, 2/C2, 3/C3, 4/C4, 5/C5, 6/C6, 7/C7, 8/C8 ].

