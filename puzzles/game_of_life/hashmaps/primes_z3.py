# Reference: The First 10,000 Primes - https://primes.utm.edu/lists/small/10000.txt

import itertools
import time
from typing import List

import z3

from hashmaps.primes import generate_primes


def z3_is_prime(x):
    y = z3.Int("y")
    return z3.Or([
        x == 2,              # 2 is the only even prime
        z3.And(
            x > 1,           # x is positive
            x % 1 == 0,      # x is int
            x % 2 != 0,      # x is odd
            z3.Not(z3.Exists([y], z3.And(
                y   > 1,     # y is positive
                y*y < x,     # y < sqrt(x)
                y % 2 != 0,  # y is odd
                x % y == 0   # y is not a divisor of x
            )))
        )
    ])


# z3_generate_primes(  2) | time =   0.0s | len =    2 | max =    3 |  [2, 3]
# z3_generate_primes(  4) | time =   0.0s | len =    4 | max =    7 |  [2, 3, 5, 7]
# z3_generate_primes(  8) | time =   0.0s | len =    8 | max =   31 |  [2, 3, 5, 7, 9, 11, 13, 31]
# z3_generate_primes( 16) | time =   0.1s | len =   16 | max =   61 |  [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 59, 61]
# z3_generate_primes( 32) | time =   0.2s | len =   32 | max =  113 |  [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 79, 89, 97, 101, 103, 107, 109, 113]
# z3_generate_primes( 64) | time =   1.1s | len =   64 | max =  293 |  [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 121, 127, 131, 137, 139, 149, 151, 157, 163, 167, 169, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 263, 271, 283, 289, 293]
# z3_generate_primes(128) | time =   9.8s | len =  128 | max = 1019 |  [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 121, 127, 131, 137, 139, 149, 151, 157, 163, 167, 169, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 289, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 361, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 529, 541, 547, 557, 563, 613, 631, 641, 643, 653, 659, 661, 673, 823, 827, 839, 859, 877, 881, 883, 887, 937, 1019]
# z3_generate_primes(256) | time = 149.0s | len =  256 | max = 2017 |  [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 43, 47, 49, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 121, 127, 131, 137, 139, 149, 151, 157, 163, 167, 169, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 289, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 361, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 529, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 841, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 961, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1049, 1051, 1061, 1063, 1069, 1103, 1109, 1117, 1123, 1151, 1153, 1163, 1171, 1201, 1213, 1217, 1223, 1249, 1259, 1301, 1303, 1307, 1319, 1321, 1361, 1367, 1369, 1373, 1399, 1409, 1423, 1451, 1453, 1459, 1471, 1499, 1511, 1523, 1549, 1553, 1559, 1567, 1601, 1607, 1609, 1613, 1619, 1657, 1663, 1667, 1669, 1699, 1709, 1721, 1723, 1753, 1759, 1801, 1811, 1823, 1849, 1861, 1867, 1873, 1901, 1907, 1913, 1949, 1951, 1973, 1999, 2003, 2011, 2017]
def z3_generate_primes(count = 100):
    """ Source: https://stackoverflow.com/questions/11619942/print-series-of-prime-numbers-in-python """
    output = []

    number = z3.Int('prime')
    solver = z3.Solver()
    solver.add([ number > 1, number % 1 == 0 ])  # is positive, is int
    solver.add([ z3_is_prime(number) ])
    solver.push()

    domain = 2
    while len(output) < count:
        solver.pop()
        solver.push()
        solver.add([ number < domain ])  # this helps prevent unsat
        solver.add(z3.And([ number != value for value in output ]))
        while solver.check() == z3.sat:
            value = solver.model()[number].as_long()
            solver.add([ number != value ])
            output.append( value )
            if len(output) >= count: break
        domain *= 2  # increment search space
    return sorted(output)


# size =  2 | combinations = 2 | time =   0.0s |  [2, 3]
# size =  3 | combinations = 3 | time =   0.0s |  [3, 5, 7]
# size =  4 | combinations = 4 | time =   0.0s |  [3, 5, 7, 11]
# size =  5 | combinations = 5 | time =   0.2s |  [5, 7, 13, 19, 23]
# size =  6 | combinations = 6 | time =   0.2s |  [3, 13, 17, 23, 29, 31]
# size =  7 | combinations = 7 | time =  14.0s |  [2, 3, 11, 19, 37, 41, 127]
# size =  8 | combinations = 8 | time = 971.7s |  [3, 29, 37, 71, 89, 101, 107, 113]
def z3_generate_summable_primes(size=50, combinations=2) -> List[int]:
    candidates = [ z3.Int(n) for n in range(size) ]
    summations = []
    for n_combinations in range(1,combinations+1):
        summations += [
            z3.Sum(group)
            for group in itertools.combinations(candidates, n_combinations)
        ]

    solver = z3.Solver()
    solver.add([ num > 1      for num in candidates ])
    solver.add([ num % 1 == 0 for num in candidates ])
    solver.add([ candidates[n] < candidates[n+1] for n in range(len(candidates)-1) ])  # sorted

    # solver.add([ z3_is_prime(num) for num in candidates ])
    # primes = z3_generate_primes(128)

    solver.add( z3.Distinct(candidates) )
    solver.add( z3.Distinct(summations) )
    solver.push()
    domain = 2
    while True:
        solver.pop()
        solver.push()

        primes = generate_primes(domain ** 2)
        solver.add([ z3.Or([ num == prime for prime in primes ])
                     for num in candidates ])
        solver.add([ num < domain for num in candidates ])

        if solver.check() == z3.sat:
            values = sorted([ solver.model()[num].as_long() for num in candidates ])
            return list(values)
        else:
            domain *= 2


### Output:
if __name__ == '__main__':

    # for n in [2,4,8,16,32,64,128]:
    #     print( f'{n:3d} primes generated in: {timeit.timeit(lambda: z3_generate_primes(n), number=10)/10*1000:7.1f} ms' )

    for size in range(2,8+1):
        for combinations in [2,size]:
            # combinations    = size
            time_start      = time.perf_counter()
            hashable_primes = z3_generate_summable_primes(size=size, combinations=combinations)
            time_taken      = time.perf_counter() - time_start
            print(f'size = {size:2d} | combinations = {combinations} | time = {time_taken:5.1f}s | ', hashable_primes)

