#!/usr/bin/env python3
import argparse
import time

from agents.AlphaBetaPlayer import AlphaBetaAreaPlayer, AlphaBetaPlayer, MinimaxPlayer
from agents.DistancePlayer import DistancePlayer, GreedyDistancePlayer
from agents.MCTS import MCTSMaximum, MCTSMaximumHeuristic, MCTSRandom, MCTSRandomHeuristic
from agents.UCT import UCTPlayer
from isolation import Agent, logger
from my_custom_player import CustomPlayer
from run_match_sync import play_sync
from sample_players import GreedyPlayer, RandomPlayer



def log_results(agents, scores, match_id, winner, start_time, args={}):
    if args.get('progress'):
        print('+' if winner == agents[0] else '-', end='', flush=True)

    logging = args.get('logging', 100)
    if ( logging  != 0 and match_id % logging == 0
      or match_id != 0 and match_id == args.get('rounds')
    ):
        total_average   = 100 * sum(scores[agents[0]]) / len(scores[agents[0]])
        running_average = 100 * sum( 2*(i+0.5)*score for i,score in enumerate(scores[agents[0]]) ) / len(scores[agents[0]])**2
        message = " match_id: {:4d} | {:3.0f}s | {:3.0f}% -> {:3.0f}% | {} vs {}" .format(
            match_id,
            time.perf_counter() - start_time,
            total_average, running_average,
            agents[0].name,
            agents[1].name,
        )
        print(message); logger.info(message)


def run_backpropagation(args):
    assert args['agent'].upper()    in TEST_AGENTS, '{} not in {}'.format(args['agent'],    TEST_AGENTS.keys())
    assert args['opponent'].upper() in TEST_AGENTS, '{} not in {}'.format(args['opponent'], TEST_AGENTS.keys())
    agent1 = TEST_AGENTS.get(args['agent'].upper())
    agent2 = TEST_AGENTS.get(args['opponent'].upper())
    if agent1.name == agent2.name:
        agent1 = Agent(agent1.agent_class, agent1.name)
        agent2 = Agent(agent2.agent_class, agent2.name+' 2')
    agents = (agent1, agent2)

    # Reset caches
    if args.get('reset'):
        for agent_idx, agent in enumerate(agents):
            if callable(getattr(agent.agent_class, 'reset', None)):
                agent.agent_class.reset()


    scores = {
        agent: []
        for agent in agents
    }
    start_time = time.perf_counter()
    match_id = 0
    while True:
        if args.get('rounds',  0) and args['rounds']  <= match_id:                         break
        if args.get('timeout', 0) and args['timeout'] <= time.perf_counter() - start_time: break

        match_id += 1
        agent_order = ( agents[(match_id)%2], agents[(match_id+1)%2] )  # reverse player order between matches
        winner, game_history, match_id = play_sync(agent_order, match_id=match_id, **args)

        winner_idx = agent_order.index(winner)
        loser      = agent_order[int(not winner_idx)]
        scores[winner] += [ 1 ]
        scores[loser]  += [ 0 ]

        for agent_idx, agent in enumerate(agent_order):
            if callable(getattr(agent.agent_class, 'backpropagate', None)):
                agent.agent_class.backpropagate(winner_idx=winner_idx, game_history=game_history)

        log_results(agents, scores, match_id, winner, start_time, args)



TEST_AGENTS = {
    "RANDOM":    Agent(RandomPlayer,         "Random"),
    "GREEDY":    Agent(GreedyPlayer,         "Greedy"),
    "DISTANCE":  Agent(DistancePlayer,       "Distance"),
    "GD":        Agent(GreedyDistancePlayer, "Greedy Distance"),
    "MINIMAX":   Agent(MinimaxPlayer,        "Minimax"),
    "ALPHABETA": Agent(AlphaBetaPlayer,      "AlphaBeta"),
    "AREA":      Agent(AlphaBetaAreaPlayer,  "AlphaBeta Area"),
    "MCM":       Agent(MCTSMaximum,          "MCTS Maximum"),
    "MCR":       Agent(MCTSRandom,           "MCTS Random"),
    "MCMH":      Agent(MCTSMaximumHeuristic, "MCTS Maximum Heuristic"),
    "MCRH":      Agent(MCTSRandomHeuristic,  "MCTS Random Heuristic"),
    "UCT":       Agent(UCTPlayer,            "UCT"),
    "SELF":      Agent(CustomPlayer,         "Custom TestAgent"),
}
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rounds',     type=int, default=0)
    parser.add_argument(      '--timeout',    type=int, default=0)    # train_mcts() global timeout
    parser.add_argument('-t', '--time_limit', type=int, default=150)  # play_sync()  timeout per round
    parser.add_argument('-a', '--agent',      type=str, default='SELF')
    parser.add_argument('-o', '--opponent',   type=str, default='SELF')
    parser.add_argument('-l', '--logging',    type=int, default=100)
    parser.add_argument(      '--progress',   action='store_true')    # show progress bat
    parser.add_argument(      '--reset',      action='store_true')
    parser.add_argument('-v', '--verbose',    action='store_true')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = argparser()
    run_backpropagation(args)
