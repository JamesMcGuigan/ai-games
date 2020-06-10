#!/usr/bin/env python3

#####
##### ./kaggle_compile.py agents/AlphaBetaPlayer.py
#####
##### 2020-06-09 03:07:02+01:00
#####
##### origin	git@github.com:JamesMcGuigan/udacity-artificial-intelligence.git (fetch)
##### origin	git@github.com:JamesMcGuigan/udacity-artificial-intelligence.git (push)
#####
##### * master           8445f19 3_Adversarial Search | bugfix: DataSavePlayer.save() check ./data/ directory exists
#####   origin/solutions f1b8475 Update scale parameter is board display function for CSP exercise
#####   remote           090f8a0 chore: git rm *.pyc
#####
##### 8445f19b4d3326ad9dcf7bcf4021f6a267ec8887
#####

#####
##### START agents/DataSavePlayer.py
#####

import atexit
import gzip
import os
import pickle
import zlib

from sample_players import BasePlayer



class DataSavePlayer(BasePlayer):
    data    = {}
    verbose = True

    def __new__(cls, *args, **kwargs):
        # noinspection PyUnresolvedReferences
        for parentclass in cls.__mro__:  # https://stackoverflow.com/questions/2611892/how-to-get-the-parents-of-a-python-class
            if cls is parentclass: continue
            if cls.data is getattr(parentclass,'data',None):
                cls.data = {}  # create a new cls.data for each class
                break
        instance = object.__new__(cls)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load()
        self.autosave()

    def autosave( self ):
        # Autosave on Ctrl-C
        atexit.unregister(self.__class__.save)
        atexit.register(self.__class__.save)

    # def __del__(self):
    #     self.save()

    @classmethod
    def filename( cls ):
        return './data/' + cls.__name__ + '.zip.pickle'

    @classmethod
    def load( cls ):
        if cls.data: return  # skip loading if the file is already in class memory
        try:
            # class data may be more upto date than the pickle file, so avoid race conditions with multiple instances
            filename   = cls.filename()
            start_time = time.perf_counter()
            with gzip.GzipFile(filename, 'rb') as file:  # reduce filesystem size
                # print("loading: "+cls.file )
                data = pickle.load(file)
                cls.data.update({ **data, **cls.data })
                if cls.verbose:
                    print("loaded: {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(
                        filename,
                        os.path.getsize(filename)/1024/1024,
                        time.perf_counter() - start_time,
                        cls.size(cls.data),
                    ))
        except (IOError, TypeError, EOFError, zlib.error) as exception:
            pass

    @classmethod
    def save( cls ):
        # cls.load()  # update any new information from the file
        if cls.data:
            filename = cls.filename()
            dirname  = os.path.dirname(filename)
            if not os.path.exists(dirname): os.mkdir(dirname)
            start_time = time.perf_counter()
            # print("saving: " + filename )
            with gzip.GzipFile(filename, 'wb') as file:  # reduce filesystem size
                pickle.dump(cls.data, file)
                if cls.verbose:
                    print("wrote:  {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(
                        filename,
                        os.path.getsize(filename)/1024/1024,
                        time.perf_counter() - start_time,
                        cls.size(cls.data),
                    ))

    @staticmethod
    def size( data ):
        return sum([
            len(value) if isinstance(key, str) and isinstance(value, dict) else 1
            for key, value in data.items()
        ])

    @classmethod
    def reset( cls ):
        cls.data = {}
        cls.save()


    ### Caching
    @classmethod
    def cache(cls, function, state, player_id, *args, **kwargs):
        hash = (player_id, state)  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.data:   cls.data[function.__name__] = {}
        if hash in cls.data[function.__name__]: return cls.data[function.__name__][hash]

        score = function(state, player_id, *args, **kwargs)
        cls.data[function.__name__][hash] = score
        return score

    @classmethod
    def cache_infinite(cls, function, state, player_id, *args, **kwargs):
        # Don't cache heuristic values, only terminal states
        hash = (player_id, state)  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.data:   cls.data[function.__name__] = {}
        if hash in cls.data[function.__name__]: return cls.data[function.__name__][hash]

        score = function(state, player_id, *args, **kwargs)
        if abs(score) == math.inf: cls.data[function.__name__][hash] = score
        return score


#####
##### END   agents/DataSavePlayer.py
#####

#####
##### START isolation/isolation.py
#####


###############################################################################
#                          DO NOT MODIFY THIS FILE                            #
###############################################################################
from enum import IntEnum
from typing import NamedTuple


# board array dimensions and bitboard size
_WIDTH = 11
_HEIGHT = 9
_SIZE = (_WIDTH + 2) * _HEIGHT - 2

# Build the prototype bitboard, which is a bitstring (e.g., 1110011100111
# is a 3x3 rectangular grid) See the isolation module readme for full details.
_BLANK_BOARD = 0
row = ((1<<_WIDTH) - 1)
for _ in range(_HEIGHT): _BLANK_BOARD = ((_BLANK_BOARD << (_WIDTH + 2)) | row)

# declare constants describing the bit-wise offsets for each cardinal direction
S, N, W, E = -_WIDTH - 2, _WIDTH + 2, 1, -1

class Action(IntEnum):
    """ The eight L-shaped steps that a knight can move in chess """
    NNE = N+N+E  # north-northeast (up, up, right)
    ENE = E+N+E  # east-northeast (right, right, up)
    ESE = E+S+E  # east-southeast (right, right, down)
    SSE = S+S+E  # south-southeast (down, down, right)
    SSW = S+S+W  # south-southwest (down, down, left)
    WSW = W+S+W  # west-southwest (left, left, down)
    WNW = W+N+W  # west-northwest (left, left, up)
    NNW = N+N+W  # north-northwest (up, up, left)

_ACTIONSET = set(Action)  # used for efficient membership testing


class Isolation(NamedTuple('Isolation', [('board', int), ('ply_count', int), ('locs', int)])):
    """ Bitboard implementation of knight's Isolation game state

    Subclassing NamedTuple makes the states (effectively) immutable
    and hashable. Using immutable states can help avoid errors that
    can arise with in-place state updates. Hashable states allow the
    state to be used as the key to a look up table.

    Attributes
    ----------
    board: int
        Bitboard representation of isolation game state. bits that are ones
        represent open cells; bits that are zeros represent blocked cells

    ply_count: int
        Cumulative count of the number of actions applied to the board

    locs: tuple
        A pair of values defining the location of each player. Default for
        each player is None while the player has not yet placed their piece
        on the board; otherwise an integer.
    """
    def __new__(cls, board=_BLANK_BOARD, ply_count=0, locs=(None, None)):
        return super(Isolation, cls).__new__(cls, board, ply_count, locs)

    def actions(self):
        """ Return a list of the legal actions in the current state

        Note that players can choose any open cell on the opening move,
        but all later moves MUST be one of the values in Actions.

        Returns
        -------
        list
            A list containing the endpoints of all legal moves for the
            active player on the board
        """
        loc = self.locs[self.player()]
        if loc is None:
            return self.liberties(loc)
        return [a for a in Action if (a + loc) >= 0 and (self.board & (1 << (a + loc)))]

    def player(self):
        """ Return the id (zero for first player, one for second player) of player
        currently holding initiative (i.e., the active player)
        """
        return self.ply_count % 2

    def result(self, action):
        """ Return the resulting game state after applying the action specified
        to the current game state.

        Note that players can choose any open cell on the opening move,
        but all later moves MUST be one of the values in Actions.

        Parameters
        ----------
        action : int
            An index indicating the next position for the active player

        Returns
        -------
        Isolation
            A new state object with the input move applied.
        """
        player_location = self.locs[self.player()]
        assert player_location is None or action in _ACTIONSET, \
            "{} is not a valid action from the set {}".format(action, list(Action))
        if player_location is None:
            player_location = 0
        player_location = int(action) + player_location
        if not (self.board & (1 << player_location)):
            raise RuntimeError("Invalid move: target cell blocked")
        # update the board to block the ending cell from the new move
        board = self.board ^ (1 << player_location)
        locs = (self.locs[0], player_location) if self.player() else (player_location, self.locs[1])
        return Isolation(board=board, ply_count=self.ply_count + 1, locs=locs)

    def terminal_test(self):
        """ Return True if either player has no legal moves, otherwise False

        Returns
        -------
        bool
            True if either player has no legal moves, otherwise False
        """
        return not (self._has_liberties(0) and self._has_liberties(1))

    def utility(self, player_id):
        """ Returns the utility of the current game state from the perspective
        of the specified player.

                    /  +infinity,   "player_id" wins
        utility =  |   -infinity,   "player_id" loses
                    \          0,    otherwise

        Parameters
        ----------
        player_id : int
            The 0-indexed id number of the player whose perspective is used
            for the utility calculation.

        Returns
        -------
        float
            The utility value of the current game state for the specified
            player. The game has a utility of +inf if the player has won,
            a value of -inf if the player has lost, and a value of 0
            otherwise.
        """
        if not self.terminal_test(): return 0
        player_id_is_active = (player_id == self.player())
        active_has_liberties = self._has_liberties(self.player())
        active_player_wins = (active_has_liberties == player_id_is_active)
        return float("inf") if active_player_wins else float("-inf")

    def liberties(self, loc):
        """ Return a list of "liberties"--open cells in the neighborhood of `loc`

        Parameters
        ----------
        loc : int
            A position on the current board to use as the anchor point for
            available liberties (i.e., open cells neighboring the anchor point)

        Returns
        -------
        list
            A list containing the position of open liberties in the
            neighborhood of the starting position
        """
        cells = range(_SIZE) if loc is None else (loc + a for a in Action)
        return [c for c in cells if c >= 0 and self.board & (1 << c)]

    def _has_liberties(self, player_id):
        """ Return True if the player has any legal moves in the given state

        See Also
        -------
            Isolation.liberties()
        """
        return any(self.liberties(self.locs[player_id]))


class DebugState(Isolation):
    """ Extend the Isolation game state class with utility methods for debugging &
    visualizing the fields in the data structure

    Examples
    --------
    >>> board = Isolation()
    >>> debug_board = DebugBoard.from_state(board)
    >>> print(debug_board.bitboard_string)
    11111111111001111111111100111111111110011111111111001111111111100111111111110011111111111
    >>> print(debug_board)

    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    |   |   |   |   |   |   |   |   |   |   |   |
    + - + - + - + - + - + - + - + - + - + - + - +
    """
    player_symbols=['1', '2']

    @staticmethod
    def from_state(gamestate): return DebugState(gamestate.board, gamestate.ply_count, gamestate.locs)

    @property
    def bitboard_string(self): return "{:b}".format(self.board)

    @classmethod
    def ind2xy(cls, ind):
        """ Convert from board index value to xy coordinates

        The coordinate frame is 0 in the bottom right corner, with x increasing
        along the columns progressing towards the left, and y increasing along
        the rows progressing towards teh top.
        """
        return (ind % (_WIDTH + 2), ind // (_WIDTH + 2))

    def __str__(self):
        """ Generate a string representation of the current game state, marking
        the location of each player and indicating which cells have been blocked,
        and which remain open.
        """
        import os
        from io import StringIO
        OPEN = " "
        CLOSED = "X"
        cell = "| {} "
        rowsep = "+ - " * _WIDTH + "+"
        out = StringIO()
        out.write(rowsep + os.linesep)

        board = self.board << 2
        for loc in range(_SIZE + 2):
            if loc > 2 and loc % (_WIDTH + 2) == 0:
                out.write("|" + os.linesep + rowsep + os.linesep)
            if loc % (_WIDTH + 2) == 0 or loc % (_WIDTH + 2) == 1:
                continue
            sym = OPEN if (board & (1 << loc)) else CLOSED
            if loc - 2 == self.locs[0]: sym = self.player_symbols[0]
            if loc - 2 == self.locs[1]: sym = self.player_symbols[1]
            out.write(cell.format(sym))
        out.write("|" + os.linesep + rowsep + os.linesep)
        return '\n'.join(l[::-1] for l in out.getvalue().split('\n')[::-1]) + os.linesep


#####
##### END   isolation/isolation.py
#####

#####
##### START agents/AlphaBetaPlayer.py
#####

import math
import random
import time
from functools import lru_cache
from itertools import chain
from operator import itemgetter

# from agents.DataSavePlayer import DataSavePlayer
# from isolation.isolation import Action



class AlphaBetaPlayer(DataSavePlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the cls.context attribute.
    **********************************************************************
    """

    verbose_depth        = False
    search_fn            = 'alphabeta'       # or 'minimax'
    search_max_depth     = 50

    heuristic_fn         = 'heuristic_liberties'  # 'heuristic_liberties' | 'heuristic_area'
    heuristic_area_depth = 4                      # 4 seems to be the best number against LibertiesPlayer
    heuristic_area_max   = len(Action) * 5        # 5 seems to be the best number against LibertiesPlayer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.precache()

    # def precache( self, depth=5 ):
    #     state = Isolation()
    #     time_start = time.perf_counter()
    #     action = self.alphabeta(state, depth=depth)  # precache for first move
    #     print( 'precache()', type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        time_start = time.perf_counter()
        action = random.choice(state.actions())
        self.queue.put( action )     # backup move incase of early timeout

        # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent
        if self.verbose_depth: print('\n'+ self.__class__.__name__.ljust(20) +' | depth:', end=' ', flush=True)
        for depth in range(1, self.search_max_depth+1):
            action, score = self.search(state, depth=depth)
            self.queue.put(action)
            if self.verbose_depth: print(depth, end=' ', flush=True)
            if abs(score) == math.inf:
                if self.verbose_depth: print(score, end=' ', flush=True)
                break  # terminate iterative deepening on inescapable victory condition
        # if self.verbose_depth: print( depth, type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )


    ### Heuristics

    def heuristic(self, state, player_id):
        if self.heuristic_fn == 'heuristic_area':
            return self.heuristic_area(state, player_id)
        if self.heuristic_fn == 'heuristic_liberties':
            return self.heuristic_liberties(state, player_id)  # won 45%
        raise NotImplementedError('cls.heuristic_fn must be in ["heuristic_area", "heuristic_liberties"] - got: ', self.heuristic_fn)


    # Its not worth persisting this cache to disk, when too large (million+) its becomes quicker to compute than lookup
    @classmethod
    @lru_cache(None, typed=True)
    def heuristic_liberties(cls, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = cls.liberties(state, own_loc)
        opp_liberties = cls.liberties(state, opp_loc)
        return len(own_liberties) - len(opp_liberties)

    @staticmethod
    @lru_cache(None, typed=True)
    def liberties( state, cell ):
        """add a @lru_cache around this function"""
        return state.liberties(cell)


    @classmethod
    def heuristic_area(cls, state, player_id):
        """ Persistent caching to disk """
        return cls.cache(cls._heuristic_area, state, player_id)
    @classmethod
    def _heuristic_area(self, state, player_id):
        own_loc  = state.locs[player_id]
        opp_loc  = state.locs[1 - player_id]
        own_area = self.count_area_liberties(state, own_loc)
        opp_area = self.count_area_liberties(state, opp_loc)
        return own_area - opp_area

    @classmethod
    @lru_cache(None, typed=True)  # depth > 1 exceeds 150ms timeout (without caching)
    def count_area_liberties( cls, state, start_loc ):
        depth     = cls.heuristic_area_depth
        max_area  = cls.heuristic_area_max

        area      = set()
        frontier  = { start_loc }
        seen      = set()
        while len(frontier) and len(area) < max_area and depth > 0:
            seen     |= frontier
            frontier |= set(chain(*[ cls.liberties(state, cell) for cell in frontier ]))
            area     |= frontier
            frontier -= seen
            depth    -= 1
        return len(area)



    ### Search

    def search( self, state, depth ):
        if self.search_fn == 'minimax':
            return self.minimax(state, depth)
        if self.search_fn == 'alphabeta':
            return self.alphabeta(state, depth)
        raise NotImplementedError('cls.search_fn must be in ["minimax", "alphabeta"] - got: ', self.search_fn)


    ### Search: Minmax

    def minimax( self, state, depth ):
        actions = state.actions()
        scores  = [
            self.minimax_min_value(state.result(action), player_id=self.player_id, depth=depth-1)
            for action in actions
        ]
        action, score = max(zip(actions, scores), key=itemgetter(1))
        return action, score

    def minimax_min_value(self, state, player_id, depth):
        return self.cache_infinite(self._minimax_min_value, state, player_id, depth)
    def _minimax_min_value( self, state, player_id, depth ):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        scores = [
            self.minimax_max_value(state.result(action), player_id, depth - 1)
            for action in state.actions()
        ]
        return min(scores) if len(scores) else math.inf

    def minimax_max_value(self, state, player_id, depth):
        return self.cache_infinite(self._minimax_max_value, state, player_id, depth)
    def _minimax_max_value( self, state, player_id, depth ):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        scores = [
            self.minimax_min_value(state.result(action), player_id, depth - 1)
            for action in state.actions()
        ]
        return max(scores) if len(scores) else -math.inf


    ### Search: AlphaBeta

    def alphabeta( self, state, depth ):
        actions = state.actions()
        scores  = [
            self.alphabeta_min_value(state.result(action), player_id=self.player_id, depth=depth-1)
            for action in actions
        ]
        action, score = max(zip(actions, scores), key=itemgetter(1))
        return action, score

    def alphabeta_min_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        return self.cache_infinite(self._alphabeta_min_value, state, player_id, depth, alpha, beta)
    def _alphabeta_min_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        score = math.inf
        for action in state.actions():
            result    = state.result(action)
            score     = min(score, self.alphabeta_max_value(result, player_id, depth-1, alpha, beta))
            if score <= alpha: return score
            beta      = min(beta,score)
        return score

    def alphabeta_max_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        return self.cache_infinite(self._alphabeta_max_value, state, player_id, depth, alpha, beta)
    def _alphabeta_max_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        score = -math.inf
        for action in state.actions():
            result    = state.result(action)
            score     = max(score, self.alphabeta_min_value(result, player_id, depth-1, alpha, beta))
            if score >= beta: return score
            alpha     = max(alpha, score)
        return score



class AlphaBetaAreaPlayer(AlphaBetaPlayer):
    heuristic_fn = 'heuristic_area'  # or 'heuristic_liberties'


# This matches the specs of the course implemention - but this version is better performance optimized
class MinimaxPlayer(AlphaBetaPlayer):
    verbose          = False
    search_fn        = 'minimax'              # or 'alphabeta'
    heuristic_fn     = 'heuristic_liberties'  # 'heuristic_liberties' | 'heuristic_area'
    search_max_depth = 3



# CustomPlayer is the agent exported to the submission
class CustomPlayer(AlphaBetaAreaPlayer):
    pass


#####
##### END   agents/AlphaBetaPlayer.py
#####

#####
##### ./kaggle_compile.py agents/AlphaBetaPlayer.py
#####
##### 2020-06-09 03:07:02+01:00
#####
##### origin	git@github.com:JamesMcGuigan/udacity-artificial-intelligence.git (fetch)
##### origin	git@github.com:JamesMcGuigan/udacity-artificial-intelligence.git (push)
#####
##### * master           8445f19 3_Adversarial Search | bugfix: DataSavePlayer.save() check ./data/ directory exists
#####   origin/solutions f1b8475 Update scale parameter is board display function for CSP exercise
#####   remote           090f8a0 chore: git rm *.pyc
#####
##### 8445f19b4d3326ad9dcf7bcf4021f6a267ec8887
#####
