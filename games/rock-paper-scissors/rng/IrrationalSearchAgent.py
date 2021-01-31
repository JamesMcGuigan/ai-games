# %%writefile main.py
# Source: https://www.kaggle.com/jamesmcguigan/random-seed-search-irrational-search-agent/
# Source: https://github.com/JamesMcGuigan/ai-games/blob/master/games/rock-paper-scissors/rng/IrrationalSearchAgent.py
from typing import List, Tuple, Union

from mpmath import mp

from rng.IrrationalAgent import IrrationalAgent


class IrrationalSearchAgent(IrrationalAgent):
    """
    This attempts a Password Attack against IrrationalAgent

    Its only vulnerability is Password Attack.
    Pleased to meet you, won't you guess my name?

    There are an uncountable infinity of irrational numbers,
    but a rather limited number of useful irrationals with pronounceable names

    This will only work if the opponent has chosen to play an irrational agent
    but has chosen to play a popular named irrational
    and has also chosen the same trinary encoding algorithm

    If the opponent is not playing a known irrational sequence
    then the true Nash Equilibrium is to play a secret and unnamed irrational sequence
    """

    def __init__(self, name='irrational', irrational=None, search: List[Union[str, mp.mpf]]=None, verbose=True):
        super().__init__(name=name, irrational=irrational, verbose=verbose)

        search = {
            f'irrational#{n}': self.encode_irrational(number)
            for n, number in enumerate(search)
        } if search is not None else {}
        self.irrationals = { **self.__class__.irrationals, **search }
        self.history = {
            "action":   [],
            "opponent": []
        }


    def agent(self, obs, conf):
        """ Wrapper function for setting state """
        if obs.step > 0:
            self.history['opponent'].append(obs.lastOpponentAction)

        action = self.action(obs, conf)
        action = int(action) % conf.signs

        self.history['action'].append(action)
        return action


    def action(self, obs, conf):
        expected, irrational_name = self.search_irrationals(self.history['opponent'])
        if expected is not None:
            action   = (expected + 1) % conf.signs
            opponent = ( self.history['opponent'] or [None] )[-1]
            if self.verbose:
                print(f"{obs.step:4d} | Found Irrational: {irrational_name} | "
                      f"{opponent}{self.win_symbol()} > action {expected} -> {action}")
        else:
            action = super().action(obs, conf)  # play our own irrational number sequence
        return action


    @classmethod
    def search_irrationals(cls, sequence: List[int]) -> Tuple[int, str]:
        """
        Search through list of named irrational sequences
        if found, return the next expected number in the sequence along with the name
        """
        expected, irrational_name = None, None
        for irrational_name, irrational in cls.irrationals.items():
            if irrational[:len(sequence)] == sequence:
                expected = irrational[len(sequence)+1]
                break
        return expected, irrational_name


    ### Utility

    def win_symbol(self):
        """ Symbol representing the reward from the previous turn """
        action   = ( self.history['action']   or [None] )[-1]
        opponent = ( self.history['opponent'] or [None] )[-1]
        if isinstance(action, int) and isinstance(opponent, int):
            if action % 3 == (opponent + 1) % 3: return '+'  # win
            if action % 3 == (opponent + 0) % 3: return '|'  # draw
            if action % 3 == (opponent - 1) % 3: return '-'  # loss
        return ' '




irrational_search_instance = IrrationalSearchAgent()
def irrational_agent(obs, conf):
    return irrational_search_instance.agent(obs, conf)
