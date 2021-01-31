import pytest
from kaggle_environments import evaluate

from rng.IrrationalAgent import IrrationalAgent
from rng.IrrationalSearchAgent import IrrationalSearchAgent
import numpy as np

@pytest.mark.parametrize("name",   IrrationalSearchAgent.irrationals.keys())
@pytest.mark.parametrize("offset", [0,1,2])
def test_IrrationalSearchAgent_vs_named_irrational(name, offset):
    """ Assert we can find the full irrational sequence every time """
    episodeSteps = 1000
    results = evaluate(
        "rps",
        [
            IrrationalAgent(name=name, offset=offset),
            IrrationalSearchAgent()
        ],
        configuration={
            "episodeSteps": episodeSteps,
            # "actTimeout":   1000,
        },
    )
    assert (results[0][0] + episodeSteps/2.1) < results[0][1]



def test_IrrationalSearchAgent_vs_Irrational():
    episodeSteps = 100
    results = evaluate(
        "rps",
        [
            IrrationalAgent(),
            IrrationalSearchAgent()
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
