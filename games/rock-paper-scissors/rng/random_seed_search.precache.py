# %%time
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-random-seed-search/edit/run/52963254

from humanize import naturalsize
import numpy as np
import os

from rng.random_seed_search import cache_seeds, cache_steps, get_random, methods

# The maximum upload size is 100MB
for method in methods:
    filename = f'{method}.npy'
    if os.path.exists(filename): continue
    method_cache = np.array([
        get_random(length=cache_steps, seed=seed, method=method, use_cache=False)
        for seed in range(cache_seeds)
    ], dtype=np.int8)
    np.save(filename, method_cache)
    print(f'wrote {filename:10s} =', method_cache.shape, '=', naturalsize(os.path.getsize(filename)))
