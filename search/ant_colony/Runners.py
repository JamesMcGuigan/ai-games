import time

from search.ant_colony.AntColonySolver import AntColonySolver
from search.ant_colony.AntColonySolver import distance
from search.ant_colony.AntColonySolver import path_distance
from search.ant_colony.KMeanAntColonySolver import KmeansAntColonySolver
from search.ant_colony.util import show_path



def AntColonyRunner(cities, verbose=False, plot=False, label=None, algorithm=AntColonySolver, **kwargs):
    solver     = algorithm(cost_fn=distance, verbose=verbose, **kwargs)
    start_time = time.perf_counter()
    result     = solver.solve(cities)
    stop_time  = time.perf_counter()
    if label: kwargs = { **label, **kwargs }

    for key in ['verbose', 'plot', 'animate', 'label', 'min_time', 'max_time']:
        if key in kwargs: del kwargs[key]

    print(
        "N={:<3d} | {:5.0f} -> {:4.0f} | {:4.0f}s | ants: {:5d} | trips: {:2d} | "
        .format(len(cities), path_distance(cities), path_distance(result), (stop_time - start_time), solver.ants_used, solver.round_trips)
        + " ".join([ f"{k}={v}" for k,v in kwargs.items() ])
    )
    if plot:
        show_path(result)
    return result


def KmeansAntColonyRunner(cities, animate=False, **kwargs):
    if animate is True: animate=show_path
    kwargs = { "animate": animate, "algorithm": KmeansAntColonySolver, **kwargs }
    return AntColonyRunner(cities, **kwargs)
