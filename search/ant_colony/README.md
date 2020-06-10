# Ant Colony Optimization Algorithm with Kmeans Clustering
Solution to the Travelling Salesman Problem

## Visualization and Hyperparameter Tuning 

- https://www.kaggle.com/jamesmcguigan/ant-colony-optimization-algorithm/ 
- https://www.kaggle.com/jamesmcguigan/kmeans-ant-colony-optimization

## Ant Colony Optimization Algorithm

This notebook explains the Ant Colony Optimization Algorithm as applied to the Travelling Salesman Problem. 
- https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms 
- https://en.wikipedia.org/wiki/Travelling_salesman_problem


The Travelling Salesman Problem is a classic NP-hard problem and defined as:
> Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city and returns to the origin city?

One possible solution is take inspiration from an Ant Colony and how it used decentralized intelligence to find food:
- the colony as a whole can be considered a single decentralized organism
  - evolutionarily speaking, the queen carries the genetic information for the entire colony
  - individual ants performing the same role as individual cells in a human body
- ants communicate using pheromones and leave a pheromone trail behind them as they walk
- when no food is found, scouts will randomly wander around looking for food
- when a scout finds food, it will carry it home and leave a pheromone trail signalling that there is food in this direction
- the scout will encourage worker ants to follow it back to the food
- the worker ants will generally follow the strongest pheromone trail, but sometimes randomly wander off it
- the worker ants will also leave a pheromone trail behind them, and the more ants that follow a path, the stronger that pheromone trail will become
- once a worker ant has found food, it will follow and strengthen the pheromone trail back home
- if an ant accidentally finds a shorter path, it's round trip times will be shorter than on longer path, more round-trips mean more pheromones
- as the pheromones along the shorter path build up faster than along the longer path, it will encourage more ants to explore this path, building up the pheromones even more quickly
- eventually the network of pheromone trails will map out near-optimal solutions for the shortest paths between food locations
- when the food runs out, the ants will disperse as scouts, and the old pheromone trails will fade away
- however if the ants are not careful, they can all end up in a death spiral, all following each other in a circle but not going anywhere


# KMeans Ant Colony Optimization

The Ant Colony Optimization Algorithm can be extended using KMeans clustering, to solve for larger map sizes `N=64` to `N=256` which is explored in this Notebook.

The basic idea is that while Ant Colony Optimization is NP-Hard with a time complexity of `O(N^2)`, the algorithm is fast to solve for small map sizes. 
- KMeans is used to subdivide the map into small regionalized clusters, which can be quickly solved for small `N`
- The pheromone trail is persisted between runs, which provides a heuristic for solving large maps
- The cluster size starts off small and gradually expands
- Each new iteration presents the simplifies subproblem of how to add a few extra nodes to an existing local map
- The number of clusters is eventually reduced to 1: the original problem definition
- With a prebuilt pheromone trail, the ants can usually solve large maps within a small number of round trips
