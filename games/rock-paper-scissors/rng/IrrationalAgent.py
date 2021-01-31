# %%writefile irrational.py
# Source: https://www.kaggle.com/jamesmcguigan/random-seed-search-irrational-agent/
# Source: https://github.com/JamesMcGuigan/ai-games/blob/master/games/rock-paper-scissors/rng/IrrationalAgent.py

import re
import time
from typing import List, Union

from mpmath import mp
mp.dps = 2048  # more than 1000 as 10%+ of chars will be dropped


def encode_irrational(irrational: Union[str,mp.mpf], offset=0) -> List[int]:
    """
    Encode the irrational number into trinary
    The irrational is converted to a string, "0"s are removed
    Then each of the digits is converted to an integer % 3 and added the to sequence
    """
    if isinstance(irrational, list) and all([ 0 <= n <= 2 for n in irrational ]):
        return irrational  # prevent double encoding

    string   = re.sub('[^1-9]', '', str(irrational))
    sequence = [
        ( int(c) + int(offset) ) % 3
        for c in string
    ]
    assert len(sequence) >= 1000
    return sequence



class IrrationalAgent():
    """
    Play an fixed sequence of moves derived the digits of an irrational number

    Irrational numbers are more mathematically pure source of randomness than
    the repeating numbers used by the Mersenne Twister RNG

    This agent is pure Chaos and contains no Order capable of being exploited once the game has started

    Its only vulnerability is Password Attack.
    Pleased to meet you, won't you guess my name?

    There are an uncountable infinity of irrational numbers
    Your choice of irrational is your password
    Be irrational in your choice of irrational if want this agent to be secure
    Alternatively choose a popular irrational with an offset to attack a specific agent

    This is the true Nash Equilibrium solution to Rock Paper Scissors
    """

    irrationals = {
        name: encode_irrational(irrational)
        for name, irrational in {
            'pi':       mp.pi(),
            'phi':      mp.phi(),
            'e':        mp.e(),
            'sqrt2':    mp.sqrt(2),
            'sqrt5':    mp.sqrt(5),
            'euler':    mp.euler(),
            'catalan':  mp.catalan(),
            'apery':    mp.apery(),
            # 'khinchin':  mp.khinchin(),  # slow
            # 'glaisher':  mp.glaisher(),  # slow
            # 'mertens':   mp.mertens(),   # slow
            # 'twinprime': mp.twinprime(), # slow
        }.items()
    }


    def __init__(self, name='irrational', irrational: Union[str,mp.mpf] = None, offset=0, verbose=True):
        # Irrational numbers are pure random sequences that are immune to random seed search
        # Using name == 'irrational' causes the number to be reset each new game
        if irrational is not None and ( name == 'irrational' or name in self.irrationals.keys() ):
            name = 'secret'
        if name in self.irrationals.keys():
            irrational = self.irrationals[name]
        self.irrational = self.encode_irrational(irrational, offset=offset)

        self.name       = name
        self.offset     = offset
        self.verbose    = verbose
        self.reset()


    def reset(self):
        """
        Reset on the first turn of every new game
        This allows a single instance to be run in a loop for testing
        """
        self.history = {
            "action":   [],
            "opponent": []
        }
        if self.name == 'irrational':
            self.irrational = self.encode_irrational(None, offset=self.offset)



    def __call__(self, obs, conf):
        return self.agent(obs, conf)

    def agent(self, obs, conf):
        """ Wrapper function for setting state in child classes """

        # Generate a new history and irrational seed for each new game
        if obs.step == 0:
            self.reset()

        # Keep a record of opponent and action state
        if obs.step > 0:
            self.history['opponent'].append(obs.lastOpponentAction)

        # This is where the subclassable agent logic happens
        action = self.action(obs, conf)

        # Keep a record of opponent and action state
        self.history['action'].append(action)
        return action


    def action(self, obs, conf):
        """ Play the next digit in a fixed irrational sequence """
        action = int(self.irrational[ obs.step % len(self.irrational) ])
        action = (action + self.offset) % conf.signs
        if self.verbose:
            name = self.__class__.__name__ + ':' + self.name + (f'+{self.offset}' if self.offset else '')
            opponent = ( self.history['opponent'] or [None] )[-1]
            expected = ( action - 1 ) % 3
            print(f"{obs.step:4d} | {opponent}{self.win_symbol()} > action {action} | " +
                  f"{name}")
        return action


    @staticmethod
    def generate_secure_irrational():
        """
        Be irrational in your choice of irrational if want this agent to be secure
        """
        irrational = sum([
            mp.sqrt(n) * (time.monotonic_ns() % 2**32)
            for n in range(2, 5 + (time.monotonic_ns() % 1024))
        ])
        return irrational


    @classmethod
    def encode_irrational(cls, irrational: Union[str,mp.mpf], offset=0) -> List[int]:
        if irrational is None:
            irrational = cls.generate_secure_irrational()
        return encode_irrational(irrational, offset)



    ### Logging

    def win_symbol(self):
        """ Symbol representing the reward from the previous turn """
        action   = ( self.history['action']   or [None] )[-1]
        opponent = ( self.history['opponent'] or [None] )[-1]
        if isinstance(action, int) and isinstance(opponent, int):
            if action % 3 == (opponent + 1) % 3: return '+'  # win
            if action % 3 == (opponent + 0) % 3: return '|'  # draw
            if action % 3 == (opponent - 1) % 3: return '-'  # loss
        return ' '


irrational_instance = IrrationalAgent(name='pi', offset=0)
def irrational_agent(obs, conf):
    return irrational_instance.agent(obs, conf)
