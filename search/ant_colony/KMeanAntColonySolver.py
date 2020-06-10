# Source: https://www.kaggle.com/jamesmcguigan/kmeans-ant-colony-optimization

import math
import random
import time
from itertools import chain
from itertools import combinations
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from search.ant_colony.AntColonySolver import AntColonySolver
from search.ant_colony.AntColonySolver import path_distance



class KmeansAntColonySolver(AntColonySolver):
    def __init__(self,
                 animate: Callable=None,

                 cluster_factor=1.05,           # Multiple for subdividing the problem
                 random_factor=1,               # Create random subgroups - this doesn't work
                 distance_power_multiple=1.5,   # Increase distance_power before final solve
                 intercity_merge=0,             # For each pair of clusters, find a loop between the N nearest neighbours of each
                 intercity_loop=0,              # Construct loops between clusters using N nearest neighbours for each cluster
                 intercity_random=0,            # Construct loops between random members of each cluster

                 start_smell=2,
                 start_smell_normalization=0.5,
                 min_round_trips=2,
                 max_round_trips=0,             # performance shortcut
                 best_path_smell=1.25,          # 2*1 + 1.25*0.5 work best
                 best_path_smell_multiple=0.5,  # Increase best_path_smell before final solve
                 min_clusters=2,
                 min_cluster_size=3,

                 **kwargs
    ):
        self.min_clusters         = min_clusters
        self.min_cluster_size     = min_cluster_size


        self.animate              = animate
        self.intercity_merge      = intercity_merge
        self.intercity_loop       = intercity_loop
        self.intercity_random     = intercity_random
        self.cluster_factor       = cluster_factor
        self.random_factor        = random_factor
        self.distance_power_multiple   = distance_power_multiple
        self.best_path_smell_multiple  = best_path_smell_multiple
        self.start_smell_normalization = start_smell_normalization


        self.kwargs = {
            "start_smell":     start_smell,
            "min_round_trips": min_round_trips,
            "max_round_trips": max_round_trips,
            "best_path_smell": best_path_smell,
            **kwargs
        }
        super().__init__(**self.kwargs)

        ### Heuristic Exports
        self.ants_used   = 0
        self.epochs_used = 0
        self.round_trips = 0


    @staticmethod
    def get_numeric_path( problem_path: List[Any] ) -> List[Tuple[int, int]]:
        # KMeans requires: List[Tuple[int,int]]
        numeric_path = list(problem_path)
        try:
            if isinstance(numeric_path, dict):      numeric_path = list(numeric_path.values())            # if path == {"Name": (x,y)}
            if isinstance(numeric_path[0][0], str): numeric_path = [ item[1] for item in numeric_path ]   # if path == ("Name", (x,y))
        except: pass
        return numeric_path


    @staticmethod
    def group_by_random( problem_path: List[Any], n_clusters ) -> List[Any]:
        clusters = [
            random.sample(problem_path, math.ceil(len(problem_path) / n_clusters) )
            for _ in range(int(n_clusters))
        ]
        return clusters


    def group_by_kmeans(self, problem_path: List[Any], n_clusters) -> List[List[Any]]:
        if n_clusters == 1: return [ problem_path ]

        # Group the cities into KMeans cluster groups of increasing size
        numeric_path = self.get_numeric_path(problem_path)
        cluster_ids  = KMeans(n_clusters=n_clusters).fit_predict(numeric_path)
        clusters = [
            list({ problem_path[n] for n in range(len(problem_path)) if cluster_ids[n] == cluster_id })
            for cluster_id in np.unique(cluster_ids)
        ]
        return clusters


    def centroid(self, problem_path: List[Tuple[Any]]) -> Tuple[int,int]:
        numeric_path = self.get_numeric_path(problem_path)
        return tuple(np.median(numeric_path, axis=0))

        # Returns the two nearest neighbours to the centroid for each cluster
    def nearest_neighbors(self, clusters: List[List[Any]], n_neighbors=2) -> List[List[Any]]:
        center_point = self.centroid(list(chain(*clusters)))
        clusters_of_nearest = []
        for cluster in clusters:
            numeric_path   = self.get_numeric_path(cluster)
            nn             = NearestNeighbors(n_neighbors).fit(numeric_path)
            dist, indicies = nn.kneighbors([center_point])  # inputs and outputs are both arrays
            clusters_of_nearest.append([ cluster[i] for i in indicies[0] ])
        return clusters_of_nearest


    def normalize_pheromones(self, norm: float=None):
        norm = norm or self.start_smell
        # mean = np.mean(list(chain(*[ d.values() for d in self.pheromones.values() ])))
        for source in self.pheromones.keys():
            for dest in self.pheromones.keys():
                self.pheromones[source][dest] *= norm
                self.pheromones[source][dest] += norm * self.start_smell_normalization


    def solve(self,
              problem_path: List[Any],
              restart=True,
    ) -> List[Tuple[int,int]]:
        # Initialize the Solver - preserve the pheromone trail between runs
        self.solve_initialize(problem_path)

        # Break the Travelling Salesman problem down into local clusters of nodes, as detected by KMeans
        # Iteratively decrease the number of clusters, until we are back at the starting problem

        n_clusters = int( len(problem_path) / ( self.cluster_factor * self.random_factor ) )
        random_clusters = self.group_by_random(problem_path, self.random_factor)

        results_plot = {}
        while n_clusters > self.min_clusters:

            results_plot[n_clusters] = []
            results_plot[f"{n_clusters}_loop"]  = []
            results_plot[f"{n_clusters}_merge"] = []

            for random_cluster in random_clusters:
                kmeans_clusters = self.group_by_kmeans(random_cluster, int(n_clusters))
                kmeans_clusters = [ cluster for cluster in kmeans_clusters if len(cluster) >= self.min_cluster_size ]

                # Break the map down into kmeans subclusters and create a heuristic pheromone trail
                for kmeans_cluster in kmeans_clusters:
                    if len(kmeans_cluster) < self.min_cluster_size: continue
                    results = self.solve_subproblem(kmeans_cluster, restart=False)
                    results_plot[n_clusters] += [ results ]

                if len(kmeans_clusters) <= 1: continue  # Can't do intercity with a single cluster

                # Construct a loop between clusters, using the N closest members to the centroid from each cluster
                if self.intercity_loop:
                    intercity = self.nearest_neighbors(kmeans_clusters, self.intercity_loop)
                    intercity = list(chain(*intercity))
                    results   = self.solve_subproblem(intercity, restart=False)
                    results_plot[f"{n_clusters}_loop"] += [ results ]

                if self.intercity_random:
                    intercity = [ random.sample(cluster, max(self.intercity_random, len(cluster)-1)) for cluster in kmeans_clusters ]
                    intercity = list(chain(*intercity))
                    results   = self.solve_subproblem(intercity, restart=False)
                    results_plot[f"{n_clusters}_loop"] += [ results ]

                # For each pair of clusters, find the optimal path to join them using their N nearest neighbours
                if self.intercity_merge:
                    for clusters in combinations(kmeans_clusters, 2):
                        intercity = self.nearest_neighbors(clusters, self.intercity_merge)
                        intercity = list(chain(*intercity))
                        results   = self.solve_subproblem(intercity, restart=False)
                        results_plot[f"{n_clusters}_merge"] += [ results ]
                        # self.normalize_pheromones()
            n_clusters = int( (n_clusters) // ( self.cluster_factor * self.random_factor ) )

        # Display the growth of clusters
        if callable(self.animate):
            self.animate_results(results_plot, problem_path)

        # Now solve the original problem
        for key, value in self.kwargs.items():
            if hasattr(self, key): setattr(self, key, value)

        self.normalize_pheromones()
        self.distance_power  *= self.distance_power_multiple
        self.best_path_smell *= self.best_path_smell_multiple
        self.round_trips = 0
        self.ant_count   = 4 * len(problem_path)
        #self.min_ants    = self.ants_used + len(problem_path) ** 2 / 2
        self.max_ants    = self.ants_used + len(problem_path) ** 2 * 2
        result = super().solve(problem_path)

        if callable(self.animate):
            plt.figure()
            self.animate(result)

        return result


    def solve_subproblem(self, problem_path: List[Any], restart=True) -> List[Tuple[int,int]]:
        verbose = self.verbose
        self.round_trips = 0
        self.ant_count   = 4 * len(problem_path)
        #self.min_ants    = 0 # len(problem_path) ** 2 / 2
        #self.max_ants    = 0 # self.ants_used + len(problem_path) ** 2

        time_start    = time.perf_counter()
        self.verbose  = False
        result        = super().solve(problem_path, restart=False)
        # self.normalize_pheromones_path(problem_path, 10000)
        self.verbose  = verbose
        if self.verbose:
            print(
                f'solve({len(problem_path)})', path_distance(problem_path), '->', path_distance(result),
                { "ant_count": self.ant_count, "ants_used": self.ants_used, "round_trips": self.round_trips,  "time": round(time.perf_counter() - time_start, 1) }
            )
        return result


    def animate_results(self, results_plot: Dict[int, List[Any]], problem_path: List=None) -> None:
        results_plot = { k:v for k,v in results_plot.items() if len(v) }  # remove empty results
        if not len(results_plot):      return
        if not callable(self.animate): return
        if problem_path is None: problem_path = []

        grid_cols = max(4, math.ceil(np.sqrt(len(results_plot))))
        grid_cols = min(grid_cols,len(results_plot))
        grid_rows = math.ceil(len(results_plot)/grid_cols)
        grid_size = ( grid_rows, grid_cols )
        figure, axes = plt.subplots(*grid_size, figsize=(grid_size[0]*10, grid_size[1]*10))
        plt.tight_layout(pad=5)
        try:
            for ax in chain(*axes): ax.axis('off')
        except: pass

        for index, N in enumerate(results_plot.keys()):
            plt.subplot(*grid_size, index+1)
            # unique_lengths = list(np.unique(list(map(len,results_plot[N]))))
            plt.title(f'{len(problem_path)}/{N} = {len(results_plot[N])} clusters')
            for results in results_plot[N]:
                self.animate(results)
        #plt.close(figure)
