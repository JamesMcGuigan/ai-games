# %%writefile test_irrational.py
import numpy as np
import pytest
from kaggle_environments import evaluate

from rng.IrrationalAgent import IrrationalAgent
from rng.IrrationalSearchAgent import IrrationalSearchAgent


def test_Irrational_new_seed_each_game():
    """ Test we can rerun a single instance of IrrationalAgent
        and that it will generate a new irrational number for each game """

    episodeSteps = 10
    results = evaluate(
        "rps",
        [
            IrrationalAgent(),
            IrrationalAgent()
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,
        },
        num_episodes=100,
        debug=True
    )
    results = np.array(results).reshape((-1,2))
    assert len(np.unique(results[:,0])) >= 3, results
    assert len(np.unique(results[:,1])) >= 3, results



@pytest.mark.parametrize("name",   IrrationalSearchAgent.irrationals.keys())
@pytest.mark.parametrize("offset", [0,1,2])
def test_Irrational_vs_offset(name, offset):
    """ Assert we can find the full irrational sequence every time """
    episodeSteps = 1000
    results = evaluate(
        "rps",
        [
            IrrationalAgent(name=name, offset=offset),
            IrrationalAgent(name=name, offset=offset+1)
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,
        },
        debug=True
    )
    assert (results[0][0] + episodeSteps/2.1) < results[0][1], results



def test_Irrational_vs_Irrational():
    episodeSteps = 100

    results = evaluate(
        "rps",
        [
            IrrationalAgent(),
            IrrationalAgent()
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,
        },
        num_episodes=100,
        debug=True,
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

    assert len(results[ results == None ]) == 0    # No errored matches
    assert np.abs(totals[0]) < 0.2 * episodeSteps  # totals are within 20%
    assert np.abs(totals[1]) < 0.2 * episodeSteps  # totals are within 20%
    assert np.abs(std[0])    < 0.2 * episodeSteps  # std  within 20%
    assert np.abs(std[1])    < 0.2 * episodeSteps  # std  within 20%
