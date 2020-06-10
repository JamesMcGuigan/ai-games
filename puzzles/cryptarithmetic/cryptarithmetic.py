#!/usr/bin/env python3

import re
import time

from z3 import *



def cryptarithmetic(input: str, limit=None, unique=True):
    start_time  = time.perf_counter()
    solver      = Solver()
    token_words = re.findall(r'\b[a-zA-Z]\w*\b', input)  # words must start with a letter

    letters = { l: Int(l) for l in list("".join(token_words)) }
    words   = { w: Int(w) for w in list(token_words)          }

    # Constraint: convert letters to numbers
    for l,s in letters.items(): solver.add(0 <= s, s <= 9)

    # Constraint: letters must be unique (optional)
    if unique and len(letters) <= 10:
        solver.add(Distinct(*letters.values()))

    # Constraint: words must be unique
    solver.add(Distinct(*words.values()))

    # Constraint: first letter of words must not be zero
    for word in words.keys():
        solver.add( letters[word[0]] != 0 )

        # Constraint: convert words to decimal values
    for word, word_symbol in words.items():
        solver.add(word_symbol == Sum(*[
            letter_symbol * 10**index
            for index,letter_symbol in enumerate(reversed([
                letters[l] for l in list(word)
                ]))
            ]))

    # Constraint: problem definition as defined by input
    solver.add(eval(input, None, words))

    solutions = []
    print(input)
    while str(solver.check()) == 'sat':
        solutions.append({ str(s): solver.model()[s] for w,s in words.items() })
        print(solutions[-1])
        solver.add(Or(*[ s != solver.model()[s] for w,s in words.items() ]))
        if limit and len(solutions) >= limit: break

    run_time = round(time.perf_counter() - start_time, 1)
    print(f'== {len(solutions)} solutions found in {run_time}s ==\n')
    return solutions



if __name__ == '__main__':
    print('## Cryptarithmetic Addition and Subtraction')
    cryptarithmetic('XY - X == YX')
    cryptarithmetic('TWO + TWO == FOUR')
    cryptarithmetic('EIGHT - FOUR == FOUR', limit=4)

    print()
    print('## Cryptarithmetic Multiplication and Division')
    cryptarithmetic('ONE * TWO == THREE', limit=1)
    cryptarithmetic('X / Y == 2')  # Division by 2 is rounded
    cryptarithmetic("Y == A * X + B")
    cryptarithmetic('( FOUR - TWO ) * ( NINE - ONE ) + TWO == EIGHTEEN', limit=3)

    print()
    print('## Cryptarithmetic Powers')
    cryptarithmetic("A**2 + B**2 == C**2",  unique=False)
    cryptarithmetic("A**2 + B**2 == CD**2", unique=False)

    print()
    print('## Cryptarithmetic Challenges ##')
    print('- https://www.reddit.com/r/dailyprogrammer/comments/7p5p2o/20180108_challenge_346_easy_cryptarithmetic_solver/')
    challenges = [
        "WHAT + WAS + THY == CAUSE",
        "HIS + HORSE + IS == SLAIN",
        "HERE + SHE == COMES",
        "FOR + LACK + OF == TREAD",
        "I + WILL + PAY + THE == THEFT",
        "TEN + HERONS + REST + NEAR + NORTH + SEA + SHORE + AS + TAN + TERNS + SOAR + TO + ENTER + THERE + AS + HERONS + NEST + ON + STONES + AT + SHORE + THREE + STARS + ARE + SEEN + TERN + SNORES + ARE + NEAR == SEVVOTH",
        "SO + MANY + MORE + MEN + SEEM + TO + SAY + THAT + THEY + MAY + SOON + TRY + TO + STAY + AT + HOME +  SO + AS + TO + SEE + OR + HEAR + THE + SAME + ONE + MAN + TRY + TO + MEET + THE + TEAM + ON + THE + MOON + AS + HE + HAS + AT + THE + OTHER + TEN == TESTS",
        "THIS + A + FIRE + THEREFORE + FOR + ALL + HISTORIES + I + TELL + A + TALE + THAT + FALSIFIES + ITS + TITLE + TIS + A + LIE + THE + TALE + OF + THE + LAST + FIRE + HORSES + LATE + AFTER + THE + FIRST + FATHERS + FORESEE + THE + HORRORS + THE + LAST + FREE + TROLL + TERRIFIES + THE + HORSES + OF + FIRE + THE + TROLL + RESTS + AT + THE + HOLE + OF + LOSSES + IT + IS + THERE + THAT + SHE + STORES + ROLES + OF + LEATHERS + AFTER + SHE + SATISFIES + HER + HATE + OFF + THOSE + FEARS + A + TASTE + RISES + AS + SHE + HEARS + THE + LEAST + FAR + HORSE + THOSE + FAST + HORSES + THAT + FIRST + HEAR + THE + TROLL + FLEE + OFF + TO + THE + FOREST + THE + HORSES + THAT + ALERTS + RAISE + THE + STARES + OF + THE + OTHERS + AS + THE + TROLL + ASSAILS + AT + THE + TOTAL + SHIFT + HER + TEETH + TEAR + HOOF + OFF + TORSO + AS + THE + LAST + HORSE + FORFEITS + ITS + LIFE + THE + FIRST + FATHERS + HEAR + OF + THE + HORRORS + THEIR + FEARS + THAT + THE + FIRES + FOR + THEIR + FEASTS + ARREST + AS + THE + FIRST + FATHERS + RESETTLE + THE + LAST + OF + THE + FIRE + HORSES + THE + LAST + TROLL + HARASSES + THE + FOREST + HEART + FREE + AT + LAST + OF + THE + LAST + TROLL + ALL + OFFER + THEIR + FIRE + HEAT + TO + THE + ASSISTERS + FAR + OFF + THE + TROLL + FASTS + ITS + LIFE + SHORTER + AS + STARS + RISE + THE + HORSES + REST + SAFE + AFTER + ALL + SHARE + HOT + FISH + AS + THEIR + AFFILIATES + TAILOR + A + ROOFS + FOR + THEIR + SAFE == FORTRESSES",
    ]
    for challenge in challenges:
        cryptarithmetic(challenge, limit=1)
