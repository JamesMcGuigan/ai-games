% Source: https://stackoverflow.com/questions/4852138/prolog-programming
:- use_module(library(clpfd)).

queens(N, L) :-
    N #> 0,
    length(L, N),
    L ins 1..N,
    all_different(L),
    applyConstraintOnDescDiag(L),
    applyConstraintOnAscDiag(L),
    label(L).

applyConstraintOnDescDiag([]) :- !.
applyConstraintOnDescDiag([H|T]) :-
    insertConstraintOnDescDiag(H, T, 1),
    applyConstraintOnDescDiag(T).

insertConstraintOnDescDiag(_, [], _) :- !.
insertConstraintOnDescDiag(X, [H|T], N) :-
    H #\= X + N,
    M is N + 1,
    insertConstraintOnDescDiag(X, T, M).

applyConstraintOnAscDiag([]) :- !.
applyConstraintOnAscDiag([H|T]) :-
    insertConstraintOnAscDiag(H, T, 1),
    applyConstraintOnAscDiag(T).

insertConstraintOnAscDiag(_, [], _) :- !.
insertConstraintOnAscDiag(X, [H|T], N) :-
    H #\= X - N,
    M is N + 1,
    insertConstraintOnAscDiag(X, T, M).