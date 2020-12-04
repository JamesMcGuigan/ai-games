from numba import njit

from utils.datasets import train_df
from utils.game import life_step_3d, life_steps_delta
from utils.plot import plot_3d
from utils.util import csv_to_delta, csv_to_numpy
import numpy as np


def get_ant_starts(pheromones: np.ndarray, n_ants: int) -> np.ndarray:
    shape      = pheromones.shape
    ant_starts = (np.random.rand(n_ants,shape[0],shape[1]) < pheromones).astype(np.int8)
    return ant_starts


def get_ant_scores(ant_stops: np.ndarray, stop: np.ndarray) -> np.ndarray:
    ant_scores = np.mean(ant_stops.reshape(-1,625) == stop.reshape(625), axis=1)
    return ant_scores

# @njit
def get_ant_rewards(ant_starts: np.ndarray, ant_stops: np.ndarray, stop: np.ndarray, delta: int) -> np.ndarray:
    ant_rewards = np.array([
        get_ant_reward(ant_starts[n], ant_stops[n], stop, delta)
        for n in range(len(ant_starts))
    ], dtype=np.float32)
    
    if ant_rewards.ndim == 3:
        ant_rewards = np.mean( ant_rewards, axis=0 ).reshape(stop.shape)
    return ant_rewards

@njit
def get_ant_reward(ant_start: np.ndarray, ant_stop: np.ndarray, stop: np.ndarray, delta: int, power=1.25) -> np.ndarray:
    ant_reward = np.zeros(stop.shape, dtype=np.float32)
    for x in range(stop.shape[0]):
        for y in range(stop.shape[1]):
            score = 0
            count = 0
            for dx in range(-delta*2, delta*2+1):
                for dy in range(-delta*2, delta*2+1):
                    ix = (x + dx) % stop.shape[0]
                    iy = (y + dy) % stop.shape[1]
                    count += 1
                    if ant_stop[ix,iy] == stop[ix,iy]:
                        score += 1

            if ant_start[x,y] == 1:
                ant_reward[x,y] = (score / count) ** power
            else:
                ant_reward[x,y] = 1 - (score / count) ** power
    return ant_reward


def update_pheromones(pheromones: np.ndarray, ant_rewards: np.ndarray) -> np.ndarray:
    pheromones = (pheromones - 0.5) + (ant_rewards - 0.5) + 0.5
    return pheromones


def ant_colony_solver(stop: np.ndarray, delta: int, n_ants: int, iterations: int = 10, plot=False) -> np.ndarray:
    pheromones  = 0.5 * np.ones(stop.shape, dtype=np.float32)
    best_board  = np.zeros(stop.shape, dtype=np.int8)
    best_score  = 0
    solution_3d = []
    for n in range(iterations):
        ant_starts = get_ant_starts(pheromones, n_ants)
        ant_stops  = life_steps_delta(ant_starts, delta)
        ant_scores = get_ant_scores(ant_stops, stop)
        if np.max(ant_scores) > best_score:
            best_score = np.max(ant_scores)
            best_board = ant_starts[ np.argmax(ant_scores) ]
            print(best_score)
            if plot: solution_3d.append(best_board)
            if best_score == 1.0: break
        ant_rewards = get_ant_rewards(ant_starts, ant_stops, stop, delta)
        pheromones  = update_pheromones(pheromones, ant_rewards)
    if plot: plot_3d(np.array(solution_3d))
    return best_board


if __name__ == '__main__':
    df    = train_df
    idx   = 0
    delta = csv_to_delta(df, idx)
    start = csv_to_numpy(df, idx, key='start')
    stop  = csv_to_numpy(df, idx, key='stop')
    solution = ant_colony_solver(stop, delta, n_ants=10, iterations=1000, plot=True)

    original_3d = life_step_3d(start, delta)
    solution_3d = life_step_3d(solution, delta)

    plot_3d(original_3d)
    plot_3d(solution_3d)
