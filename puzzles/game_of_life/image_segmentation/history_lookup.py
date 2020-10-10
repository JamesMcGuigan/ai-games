# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver
from collections import defaultdict

import pydash
from joblib import delayed
from joblib import Parallel

from hashmaps.crop import filter_crop_and_center
from hashmaps.hash_functions import hash_geometric
from image_segmentation.clusters import extract_clusters
from utils.game import life_step_3d


def get_cluster_history_lookup(boards, forward_play=10):
    """
    return history[now_hash][delta][past_hash] = {
        "start": past_cluster,
        "stop":  now_cluster,
        "delta": delta,
        "count": 1
    }
    """
    history  = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    clusters = Parallel(-1)( delayed(extract_clusters)(board)       for board in boards )
    clusters = Parallel(-1)( delayed(filter_crop_and_center)(board) for board in pydash.flatten(clusters) )
    clusters = [ cluster for cluster in clusters if cluster is not None ]
    hashes   = Parallel(-1)( delayed(hash_geometric)(board)         for board in clusters )
    clusters = { hashed: cluster for hashed, cluster in zip(hashes, clusters) }  # dedup
    for cluster in clusters.values():
        futures = life_step_3d(cluster, forward_play)
        hashes  = Parallel(-1)( delayed(hash_geometric)(future) for future in futures )
        for t in range(1, forward_play+1):
            past_cluster = futures[t]
            past_hash    = hashes[t]
            for delta in range(1,5+1):
                if t + delta >= len(futures): continue
                now_cluster = futures[t + delta]
                now_hash    = hashes[t + delta]
                if not past_hash in history[now_hash][delta]:
                    history[now_hash][delta][past_hash] = {
                        "start": past_cluster,
                        "stop":  now_cluster,
                        "delta": delta,
                        "count": 1
                    }
                else:
                    history[now_hash][delta][past_hash]['count'] += 1


    # remove defaultdict and sort by count
    history = { now_hash: { delta: dict(sorted(d2.items(), key=lambda pair: pair[1]['count'], reverse=True ))
                for delta,     d2 in d1.items()      }
                for now_hash,  d1 in history.items() }

    # Remove any past boards with less than a quarter of the frequency of the most common board
    for now_hash, d1 in history.items():
        for delta, d2 in d1.items():
            max_count = max([ values['count'] for values in d2.values() ])
            for past_hash, values in list(d2.items()):
                if values['count'] < max_count/4: del history[now_hash][delta][past_hash]
    return history
