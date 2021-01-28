from kaggle_environments import make, evaluate

from rng.random_agent_seeded import random_agent_seeded
from rng.random_seed_search import random_seed_search_agent

evaluate(
    "rps",
    [random_agent_seeded, random_seed_search_agent],
    configuration={"episodeSteps": 100}
)
