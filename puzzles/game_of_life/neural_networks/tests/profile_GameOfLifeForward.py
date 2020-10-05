# Discovery:
# neural networks in batch mode can be faster than C compiled classical code, but slower when called in a loop
# also scipy uses numba @njit under the hood
#
# number: 1 | batch_size = 1000
# life_step() - numpy         = 181.7µs
# life_step() - scipy         = 232.1µs
# life_step() - njit          = 433.5µs
# gameOfLifeForward() - loop  = 567.7µs
# gameOfLifeForward() - batch = 1807.5µs  # includes .pth file loadtimes
#
# number: 100 | batch_size = 1000
# gameOfLifeForward() - batch =  29.8µs  # faster than even @njit or scipy
# life_step() - scipy         =  35.6µs
# life_step() - njit          =  42.8µs
# life_step() - numpy         = 180.8µs
# gameOfLifeForward() - loop  = 618.3µs  # much slower, but still fast compared to an expensive function
import operator
import timeit

from neural_networks.hardcoded.GameOfLifeForward_128 import GameOfLifeForward_128
from utils.game import generate_random_boards
from utils.game import life_step_njit
from utils.game import life_step_numpy
from utils.game import life_step_scipy


def profile_GameOfLifeForward():

    model  = GameOfLifeForward_128()
    boards = generate_random_boards(1_000)
    for number in [1,10,100]:
        timings = {
            'gameOfLifeForward() - batch': timeit.timeit(lambda:   model(boards),                                number=number),
            'gameOfLifeForward() - loop':  timeit.timeit(lambda: [ model(board)           for board in boards ], number=number),
            'life_step() - njit':          timeit.timeit(lambda: [ life_step_njit(board)  for board in boards ], number=number),
            'life_step() - numpy':         timeit.timeit(lambda: [ life_step_numpy(board) for board in boards ], number=number),
            'life_step() - scipy':         timeit.timeit(lambda: [ life_step_scipy(board) for board in boards ], number=number),
        }
        print(f'number: {number} | batch_size = {len(boards)}')
        for key, value in sorted(timings.items(), key=operator.itemgetter(1)):
            print(f'{key:27s} = {value/number/len(boards) * 1_000_000:5.1f}µs')
        print()



if __name__ == '__main__':
    profile_GameOfLifeForward()
