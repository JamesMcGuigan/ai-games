# %%writefile test_RandomSeedSearch.py
import numpy as np
import pytest
from kaggle_environments import evaluate

from rng.IrrationalAgent import IrrationalAgent
from rng.IrrationalSearchAgent import IrrationalSearchAgent
from rng.random_agent_seeded import random_agent_seeded
from rng.RandomSeedSearch import RandomSeedSearch


@pytest.mark.parametrize("name",   IrrationalSearchAgent.irrationals.keys())
@pytest.mark.parametrize("offset", [0,1,2])
def test_RandomSeedSearch_vs_named_irrational(name, offset):
    """ Assert we can find the full irrational sequence every time """
    assert len(RandomSeedSearch.cache)  # check cache filepath can be found
    episodeSteps = 100
    results = evaluate(
        "rps",
        [
            IrrationalAgent(name=name, offset=offset),
            RandomSeedSearch()
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,
        },
        # debug=True  # pull request
    )
    assert (results[0][0] + episodeSteps/2.1) < results[0][1]


def test_RandomSeedSearch_vs_seeded_rng():
    """ Assert we can find the full irrational sequence every time """
    assert len(RandomSeedSearch.cache)  # check cache filepath can be found
    episodeSteps = 100
    results = evaluate(
        "rps",
        [
            random_agent_seeded,
            RandomSeedSearch()
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,
        },
        # debug=True  # pull request
    )
    assert (results[0][0] + episodeSteps/2.1) < results[0][1], results



def test_RandomSeedSearch_vs_Irrational():
    """ Show we don't have a statistical advantage inside the opening book vs irrational """
    episodeSteps = RandomSeedSearch.cache_steps * 2

    results = evaluate(
        "rps",
        [
            IrrationalAgent(),
            RandomSeedSearch()
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,  # Prevent Multiprocessing TimeoutError
        },
        num_episodes=10,
        # debug=True,  # pull request
    )
    results = np.array(results).reshape((-1,2))
    totals  = np.mean(results, axis=0)
    std     = np.std(results, axis=0).round(2)
    winrate = [ np.sum(results[:,0]-20 > results[:,1]),
                np.sum(results[:,0]+20 < results[:,1]) ]

    print('results', results)
    print('totals',  totals)
    print('std',     std)
    print('winrate', winrate)

    assert len(results[ results == None ]) == 0       # No errored matches
    assert np.abs(totals[0])    < 0.2 * episodeSteps  # scores are within 20%
    assert np.abs(totals[1])    < 0.2 * episodeSteps  # scores are within 20%
    assert np.abs(std[0]) < 0.2 * episodeSteps        # std  within 20%
    assert np.abs(std[1]) < 0.2 * episodeSteps        # std  within 20%



def test_RandomSeedSearch_vs_unseeded_RNG():
    """ Show we have a statistical advantage vs RNG """
    # episodeSteps = RandomSeedSearch.cache_steps * 2

    results = evaluate(
        "rps",
        [
            "rng/random_agent_unseeded.py",
            RandomSeedSearch()
        ],
        configuration={
            # "episodeSteps": episodeSteps,
            # "actTimeout":   1000,  # Prevent Multiprocessing TimeoutError
        },
        num_episodes=1,
        # debug=True,  # pull request
    )
    results = np.array(results).reshape((-1,2))
    totals  = np.sum(results, axis=0)
    std     = np.std(results, axis=0).round(2)
    winrate = [ np.sum(results[:,0]-20 > results[:,1]),
                np.sum(results[:,0]+20 < results[:,1]) ]

    print('results', results)
    print('totals',  totals)
    print('std',     std)
    print('winrate', winrate)

    assert len(results[ results == None ]) == 0   # No errored matches
    assert winrate[0] <= winrate[1], results      # We have a winrate advantage or draw
    assert totals[0]  <  totals[1],  totals       # We have a statistical advantage
    # assert np.abs(std[0]) < 0.2 * episodeSteps  # std within 20%
    # assert np.abs(std[1]) < 0.2 * episodeSteps  # std within 20%
