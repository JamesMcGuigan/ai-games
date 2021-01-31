# %%writefile main.py
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
        (f'{name}+{offset}' if offset else name): encode_irrational(irrational, offset=offset)
        for offset in [0,1,2]
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
        # DOCS: https://mpmath.org/doc/current/functions/constants.html
        name = name or 'irrational'
        if irrational is None:
            if name == 'irrational':
                irrational = self.generate_secure_irrational()
            else:
                assert name in self.irrationals.keys()
                irrational = self.irrationals[name]

        self.name       = name
        self.offset     = offset
        self.irrational = self.encode_irrational(irrational, offset=offset)
        self.verbose    = verbose


    def __call__(self, obs, conf):
        return self.agent(obs, conf)


    def agent(self, obs, conf):
        """ Wrapper function for setting state in child classes """
        return self.action(obs, conf)


    def action(self, obs, conf):
        """ Play the next digit in a fixed irrational sequence """
        action = int(self.irrational[ obs.step % len(self.irrational) ]) % conf.signs
        if self.verbose:
            print(f'{obs.step:4d} | {self.__class__.__name__} {self.name}[{obs.step}] = {action}')
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


    @staticmethod
    def encode_irrational(irrational: Union[str,mp.mpf], offset=0) -> List[int]:
        return encode_irrational(irrational, offset)


irrational_instance = IrrationalAgent(name='pi', offset=0)
def irrational_agent(obs, conf):
    return irrational_instance.agent(obs, conf)
