#!/usr/bin/env swipl -q -s
% Socrates Logic
%
% swipl -f socrates.prolog
% ['socrates.prolog'].
% write('Is Socrates mortal?'). mortal(socrates).
% write('Who is mortal?'). mortal(X).
% is_socrates_mortal.

% trace.    % turn on  debugging
% notrace.  % turn off debugging

% Socrates
mortal(X)   :- human(X).  % All men are mortal
human(socrates).          % Socrates is human
is_socrates_mortal :- mortal(socrates), write('Socrates is mortal').


% Zeus
-mortal(X)  :- deity(X).    % all deities are not mortal
immortal(X) :- -mortal(X).  % if immortal, then not mortal

deity(zeus).
is_zeus_immortal    :- immortal(zeus), -mortal(zeus), write('Zeus is immortal').
is_socrates_a_deity :- deity(socrates), write('Socrates is immortal').


% DOCS: https://www.swi-prolog.org/pldoc/man?section=runcomp
% swipl -q -s ./socrates.prolog
:- initialization main.
main :-
    is_socrates_mortal, nl,
    is_zeus_immortal, nl
    % is_socreates_a_deity, nl,  %  Initialization goal failed
    % halt
    .
