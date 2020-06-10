# Cryptarithmetic Solver

> Verbal arithmetic, also known as alphametics, cryptarithmetic, cryptarithm or word addition, is a type of mathematical game consisting of a mathematical equation among unknown numbers, whose digits are represented by letters. The goal is to identify the value of each letter. The name can be extended to puzzles that use non-alphabetic symbols instead of letters.
>
> <img src='https://wikimedia.org/api/rest_v1/media/math/render/svg/60eeaf958fa73a6a989f00725cf7d4c3f516e929' style='text-align:left' alt="SEND + MORE = MONEY"/>
> 
> https://en.wikipedia.org/wiki/Verbal_arithmetic

This is a general purpose solver that can handle addition, subtraction, multiplication, integer division and raising to powers.

This implementation uses the z3 constraint satisfaction solver 
- https://github.com/Z3Prover/z3/wiki

Kaggle Version: 
- https://www.kaggle.com/jamesmcguigan/cryptarithmetic-solver

## Cryptarithmetic Addition and Subtraction
``` 
XY - X == YX
{'XY': 98, 'X': 9, 'YX': 89}
== 1 solutions found in 0.0s ==

TWO + TWO == FOUR
{'TWO': 734, 'FOUR': 1468}
{'TWO': 846, 'FOUR': 1692}
{'TWO': 765, 'FOUR': 1530}
{'TWO': 867, 'FOUR': 1734}
{'TWO': 836, 'FOUR': 1672}
{'TWO': 938, 'FOUR': 1876}
{'TWO': 928, 'FOUR': 1856}
== 7 solutions found in 3.1s ==

EIGHT - FOUR == FOUR
{'EIGHT': 15864, 'FOUR': 7932}
{'EIGHT': 10476, 'FOUR': 5238}
{'EIGHT': 10784, 'FOUR': 5392}
{'EIGHT': 10764, 'FOUR': 5382}
== 4 solutions found in 2.7s ==

```

## Cryptarithmetic Multiplication
```
ONE * TWO == THREE
{'ONE': 105, 'TWO': 271, 'THREE': 28455}
== 1 solutions found in 2.6s ==

X / Y == 2
{'X': 2, 'Y': 1}
{'X': 4, 'Y': 2}
{'X': 5, 'Y': 2}
{'X': 6, 'Y': 3}
{'X': 7, 'Y': 3}
{'X': 8, 'Y': 3}
{'X': 8, 'Y': 4}
{'X': 9, 'Y': 4}
== 8 solutions found in 0.4s ==

Y == A * X + B
{'Y': 7, 'A': 3, 'X': 2, 'B': 1}
{'Y': 5, 'A': 2, 'X': 1, 'B': 3}
{'Y': 9, 'A': 1, 'X': 7, 'B': 2}
{'Y': 6, 'A': 1, 'X': 4, 'B': 2}
{'Y': 7, 'A': 1, 'X': 5, 'B': 2}
{'Y': 8, 'A': 1, 'X': 6, 'B': 2}
{'Y': 5, 'A': 1, 'X': 3, 'B': 2}
{'Y': 9, 'A': 4, 'X': 2, 'B': 1}
{'Y': 8, 'A': 6, 'X': 1, 'B': 2}
{'Y': 7, 'A': 5, 'X': 1, 'B': 2}
{'Y': 9, 'A': 7, 'X': 1, 'B': 2}
{'Y': 6, 'A': 4, 'X': 1, 'B': 2}
{'Y': 5, 'A': 3, 'X': 1, 'B': 2}
{'Y': 7, 'A': 2, 'X': 3, 'B': 1}
{'Y': 8, 'A': 1, 'X': 3, 'B': 5}
{'Y': 7, 'A': 1, 'X': 3, 'B': 4}
{'Y': 7, 'A': 1, 'X': 4, 'B': 3}
{'Y': 8, 'A': 1, 'X': 5, 'B': 3}
{'Y': 9, 'A': 1, 'X': 5, 'B': 4}
{'Y': 9, 'A': 1, 'X': 4, 'B': 5}
{'Y': 9, 'A': 1, 'X': 6, 'B': 3}
{'Y': 9, 'A': 1, 'X': 2, 'B': 7}
{'Y': 9, 'A': 1, 'X': 3, 'B': 6}
{'Y': 8, 'A': 1, 'X': 2, 'B': 6}
{'Y': 6, 'A': 1, 'X': 2, 'B': 4}
{'Y': 7, 'A': 1, 'X': 2, 'B': 5}
{'Y': 5, 'A': 1, 'X': 2, 'B': 3}
{'Y': 6, 'A': 2, 'X': 1, 'B': 4}
{'Y': 7, 'A': 2, 'X': 1, 'B': 5}
{'Y': 8, 'A': 5, 'X': 1, 'B': 3}
{'Y': 7, 'A': 4, 'X': 1, 'B': 3}
{'Y': 9, 'A': 6, 'X': 1, 'B': 3}
{'Y': 9, 'A': 3, 'X': 1, 'B': 6}
{'Y': 9, 'A': 2, 'X': 1, 'B': 7}
{'Y': 9, 'A': 5, 'X': 1, 'B': 4}
{'Y': 9, 'A': 4, 'X': 1, 'B': 5}
{'Y': 8, 'A': 2, 'X': 1, 'B': 6}
{'Y': 8, 'A': 3, 'X': 1, 'B': 5}
{'Y': 7, 'A': 3, 'X': 1, 'B': 4}
{'Y': 9, 'A': 2, 'X': 4, 'B': 1}
== 40 solutions found in 0.9s ==

( FOUR - TWO ) * ( NINE - ONE ) + TWO == EIGHTEEN
{'FOUR': 7297, 'TWO': 612, 'NINE': 2521, 'ONE': 221, 'EIGHTEEN': 15376112}
{'FOUR': 1904, 'TWO': 119, 'NINE': 9591, 'ONE': 991, 'EIGHTEEN': 15351119}
{'FOUR': 1974, 'TWO': 619, 'NINE': 9091, 'ONE': 991, 'EIGHTEEN': 10976119}
== 3 solutions found in 3228.3s ==
```

## Cryptarithmetic Powers
```
A**2 + B**2 == C**2
{'A': 3, 'B': 4, 'C': 5}
{'A': 4, 'B': 3, 'C': 5}
== 2 solutions found in 0.7s ==

A**2 + B**2 == CD**2
== 0 solutions found in 3.2s ==

```  

## Cryptarithmetic Challenges 
- https://www.reddit.com/r/dailyprogrammer/comments/7p5p2o/20180108_challenge_346_easy_cryptarithmetic_solver/
```
WHAT + WAS + THY == CAUSE
{'WHAT': 9206, 'WAS': 903, 'THY': 625, 'CAUSE': 10734}
== 1 solutions found in 1.4s ==

HIS + HORSE + IS == SLAIN
{'HIS': 354, 'HORSE': 39748, 'IS': 54, 'SLAIN': 40156}
== 1 solutions found in 1.1s ==

HERE + SHE == COMES
{'HERE': 9454, 'SHE': 894, 'COMES': 10348}
== 1 solutions found in 0.3s ==

FOR + LACK + OF == TREAD
{'FOR': 540, 'LACK': 9678, 'OF': 45, 'TREAD': 10263}
== 1 solutions found in 2.4s ==

I + WILL + PAY + THE == THEFT
{'I': 8, 'WILL': 9833, 'PAY': 526, 'THE': 104, 'THEFT': 10471}
== 1 solutions found in 2.2s ==

TEN + HERONS + REST + NEAR + NORTH + SEA + SHORE + AS + TAN + TERNS + SOAR + TO + ENTER + THERE + AS + HERONS + NEST + ON + STONES + AT + SHORE + THREE + STARS + ARE + SEEN + TERN + SNORES + ARE + NEAR == SEVVOTH
{'TEN': 957, 'HERONS': 356471, 'REST': 6519, 'NEAR': 7526, 'NORTH': 74693, 'SEA': 152, 'SHORE': 13465, 'AS': 21, 'TAN': 927, 'TERNS': 95671, 'SOAR': 1426, 'TO': 94, 'ENTER': 57956, 'THERE': 93565, 'NEST': 7519, 'ON': 47, 'STONES': 194751, 'AT': 29, 'THREE': 93655, 'STARS': 19261, 'ARE': 265, 'SEEN': 1557, 'TERN': 9567, 'SNORES': 174651, 'SEVVOTH': 1588493}
== 1 solutions found in 802.3s ==

SO + MANY + MORE + MEN + SEEM + TO + SAY + THAT + THEY + MAY + SOON + TRY + TO + STAY + AT + HOME +  SO + AS + TO + SEE + OR + HEAR + THE + SAME + ONE + MAN + TRY + TO + MEET + THE + TEAM + ON + THE + MOON + AS + HE + HAS + AT + THE + OTHER + TEN == TESTS
{'SO': 31, 'MANY': 2764, 'MORE': 2180, 'MEN': 206, 'SEEM': 3002, 'TO': 91, 'SAY': 374, 'THAT': 9579, 'THEY': 9504, 'MAY': 274, 'SOON': 3116, 'TRY': 984, 'STAY': 3974, 'AT': 79, 'HOME': 5120, 'AS': 73, 'SEE': 300, 'OR': 18, 'HEAR': 5078, 'THE': 950, 'SAME': 3720, 'ONE': 160, 'MAN': 276, 'MEET': 2009, 'TEAM': 9072, 'ON': 16, 'MOON': 2116, 'HE': 50, 'HAS': 573, 'OTHER': 19508, 'TEN': 906, 'TESTS': 90393}
== 1 solutions found in 90.2s ==

THIS + A + FIRE + THEREFORE + FOR + ALL + HISTORIES + I + TELL + A + TALE + THAT + FALSIFIES + ITS + TITLE + TIS + A + LIE + THE + TALE + OF + THE + LAST + FIRE + HORSES + LATE + AFTER + THE + FIRST + FATHERS + FORESEE + THE + HORRORS + THE + LAST + FREE + TROLL + TERRIFIES + THE + HORSES + OF + FIRE + THE + TROLL + RESTS + AT + THE + HOLE + OF + LOSSES + IT + IS + THERE + THAT + SHE + STORES + ROLES + OF + LEATHERS + AFTER + SHE + SATISFIES + HER + HATE + OFF + THOSE + FEARS + A + TASTE + RISES + AS + SHE + HEARS + THE + LEAST + FAR + HORSE + THOSE + FAST + HORSES + THAT + FIRST + HEAR + THE + TROLL + FLEE + OFF + TO + THE + FOREST + THE + HORSES + THAT + ALERTS + RAISE + THE + STARES + OF + THE + OTHERS + AS + THE + TROLL + ASSAILS + AT + THE + TOTAL + SHIFT + HER + TEETH + TEAR + HOOF + OFF + TORSO + AS + THE + LAST + HORSE + FORFEITS + ITS + LIFE + THE + FIRST + FATHERS + HEAR + OF + THE + HORRORS + THEIR + FEARS + THAT + THE + FIRES + FOR + THEIR + FEASTS + ARREST + AS + THE + FIRST + FATHERS + RESETTLE + THE + LAST + OF + THE + FIRE + HORSES + THE + LAST + TROLL + HARASSES + THE + FOREST + HEART + FREE + AT + LAST + OF + THE + LAST + TROLL + ALL + OFFER + THEIR + FIRE + HEAT + TO + THE + ASSISTERS + FAR + OFF + THE + TROLL + FASTS + ITS + LIFE + SHORTER + AS + STARS + RISE + THE + HORSES + REST + SAFE + AFTER + ALL + SHARE + HOT + FISH + AS + THEIR + AFFILIATES + TAILOR + A + ROOFS + FOR + THEIR + SAFE == FORTRESSES
{'THIS': 9874, 'A': 1, 'FIRE': 5730, 'THEREFORE': 980305630, 'FOR': 563, 'ALL': 122, 'HISTORIES': 874963704, 'I': 7, 'TELL': 9022, 'TALE': 9120, 'THAT': 9819, 'FALSIFIES': 512475704, 'ITS': 794, 'TITLE': 97920, 'TIS': 974, 'LIE': 270, 'THE': 980, 'OF': 65, 'LAST': 2149, 'HORSES': 863404, 'LATE': 2190, 'AFTER': 15903, 'FIRST': 57349, 'FATHERS': 5198034, 'FORESEE': 5630400, 'HORRORS': 8633634, 'FREE': 5300, 'TROLL': 93622, 'TERRIFIES': 903375704, 'RESTS': 30494, 'AT': 19, 'HOLE': 8620, 'LOSSES': 264404, 'IT': 79, 'IS': 74, 'THERE': 98030, 'SHE': 480, 'STORES': 496304, 'ROLES': 36204, 'LEATHERS': 20198034, 'SATISFIES': 419745704, 'HER': 803, 'HATE': 8190, 'OFF': 655, 'THOSE': 98640, 'FEARS': 50134, 'TASTE': 91490, 'RISES': 37404, 'AS': 14, 'HEARS': 80134, 'LEAST': 20149, 'FAR': 513, 'HORSE': 86340, 'FAST': 5149, 'HEAR': 8013, 'FLEE': 5200, 'TO': 96, 'FOREST': 563049, 'ALERTS': 120394, 'RAISE': 31740, 'STARES': 491304, 'OTHERS': 698034, 'ASSAILS': 1441724, 'TOTAL': 96912, 'SHIFT': 48759, 'TEETH': 90098, 'TEAR': 9013, 'HOOF': 8665, 'TORSO': 96346, 'FORFEITS': 56350794, 'LIFE': 2750, 'THEIR': 98073, 'FIRES': 57304, 'FEASTS': 501494, 'ARREST': 133049, 'RESETTLE': 30409920, 'HARASSES': 81314404, 'HEART': 80139, 'OFFER': 65503, 'HEAT': 8019, 'ASSISTERS': 144749034, 'FASTS': 51494, 'SHORTER': 4863903, 'STARS': 49134, 'RISE': 3740, 'REST': 3049, 'SAFE': 4150, 'SHARE': 48130, 'HOT': 869, 'FISH': 5748, 'AFFILIATES': 1557271904, 'TAILOR': 917263, 'ROOFS': 36654, 'FORTRESSES': 5639304404}
== 1 solutions found in 26153.0s ==
```
