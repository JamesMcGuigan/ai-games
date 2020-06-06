import argparse
import logging
import signal
import textwrap
import time
import traceback
from copy import copy
from queue import LifoQueue
from typing import Callable, List, Tuple

from isolation import Agent, DebugState, ERR_INFO, GAME_INFO, Isolation, RESULT_INFO, Status, logger
from my_custom_player import CustomPlayer
from run_match import NUM_ROUNDS, TEST_AGENTS, TIME_LIMIT



def argparser():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run matches to test the performance of your agent against sample opponents.",
        epilog=textwrap.dedent("""\
            Example Usage:
            --------------
            - Run 40 games (10 rounds = 20 games x2 for fair matches = 40 games) against
              the greedy agent with 4 parallel processes: 

                $python run_match.py -f -r 10 -o GREEDY -p 4

            - Run 100 rounds (100 rounds = 200 games) against the minimax agent with 1 process:

                $python run_match.py -r 100
        """)
        )
    parser.add_argument(
        '-d', '--debug', action="store_true",
        help="""\
            Run the matches in debug mode, which disables multiprocessing & multithreading
            support. This may be useful for inspecting memory contents during match execution,
            however, this also prevents the isolation library from detecting timeouts and
            terminating your code.
        """
        )
    # parser.add_argument(
    #     '-f', '--fair_matches', action="store_true",
    #     help="""\
    #         Run 'fair' matches to mitigate differences caused by opening position
    #         (useful for analyzing heuristic performance).  Setting this flag doubles
    #         the number of rounds your agent will play.  (See README for details.)
    #     """
    #     )
    parser.add_argument(
        '-r', '--rounds', type=int, default=NUM_ROUNDS,
        help="""\
            Choose the number of rounds to play. Each round consists of two matches 
            so that each player has a turn as first player and one as second player.  
            This helps mitigate performance differences caused by advantages for either 
            player getting first initiative.  (Hint: this value is very low by default 
            for rapid iteration, but it should be increased significantly--to 50-100 
            or more--in order to increase the confidence in your results.
        """
        )
    parser.add_argument(
        '-o', '--opponent', type=str, default='MINIMAX', choices=list(TEST_AGENTS.keys()),
        help="""\
            Choose an agent for testing. The random and greedy agents may be useful 
            for initial testing because they run more quickly than the minimax agent.
        """
        )
    # parser.add_argument(
    #     '-p', '--processes', type=int, default=NUM_PROCS,
    #     help="""\
    #         Set the number of parallel processes to use for running matches.  WARNING:
    #         Windows users may see inconsistent performance using >1 thread.  Check the
    #         log file for time out errors and increase the time limit (add 50-100ms) if
    #         your agent performs poorly.
    #     """
    #     )
    parser.add_argument(
        '-t', '--time_limit', type=int, default=TIME_LIMIT,
        help="Set the maximum allowed time (in milliseconds) for each call to agent.get_action()."
    )
    parser.add_argument(
        '-v', '--verbose', action="store_true",
        help="Print and log actions and board postions after each turn"
    )
    args = parser.parse_args()

    logging.basicConfig(filename="matches.log", filemode="w", level=logging.DEBUG)
    logging.info(
        "Search Configuration:\n" +
        "Opponent: {}\n".format(args.opponent) +
        "Rounds: {}\n".format(args.rounds) +
        # "Fair Matches: {}\n".format(args.fair_matches) +
        "Time Limit: {}\n".format(args.time_limit) +
        # "Processes: {}\n".format(args.processes) +
        "Debug Mode: {}".format(args.debug)
    )
    return args


def call_with_timeout_ms( time_limit, function, *args, **kwargs ):
    if time_limit:
        def raise_timeout(signum, frame): raise TimeoutError
        signal.signal(signal.SIGPROF, raise_timeout)           # Register function to raise a TimeoutError on signal
        signal.setitimer(signal.ITIMER_PROF, time_limit/1000)  # Schedule the signal to be sent after time_limit in milliseconds

    try:
        output = function(*args, **kwargs)
        return output
    except TimeoutError as err:
        return TimeoutError


# Modified from Source: isolation/__init__.py:_play()
def play_sync( agents: Tuple[Agent,Agent],
               game_state = None,   # defaults to Isolation()
               time_limit = TIME_LIMIT,
               match_id   = 0,
               debug      = False,  # disables the signal timeout
               verbose    = False,  # prints an ASCII copy of the board after each turn
               callbacks: List[ Callable ] = None,
               **kwargs ):

    agents        = tuple( Agent(agent, agent.__class__.name) if not isinstance(agent, Agent) else agent for agent in agents )
    players       = tuple( a.agent_class(player_id=i) for i, a in enumerate(agents) )
    game_state    = game_state or Isolation()
    initial_state = game_state
    active_idx    = 0
    winner        = None
    loser         = None
    status        = Status.NORMAL
    game_history  = []
    callbacks     = copy(callbacks) or []

    logger.info(GAME_INFO.format(initial_state, *agents))
    while not game_state.terminal_test():
        active_idx    = game_state.player()
        active_player = players[active_idx]
        winner, loser = agents[1 - active_idx], agents[active_idx]  # any problems during get_action means the active player loses

        action = None
        active_player.queue = LifoQueue()  # we don't need a TimeoutQueue here
        try:
            if time_limit == 0 or debug:
                active_player.get_action(game_state)
                action = active_player.queue.get(block=False)  # raises Empty if agent did not respond
            else:
                # increment timeout 2x before throwing exception - MinimaxAgent occasionally takes longer than 150ms
                for i in [1,2]:
                    exception = call_with_timeout_ms(i * time_limit, active_player.get_action, game_state)
                    if not active_player.queue.empty():
                        action = active_player.queue.get(block=False)  # raises Empty if agent did not respond
                        break                                          # accept answer generated after minimum timeout
                if action is None and exception == TimeoutError:
                    print(active_player)
                    raise TimeoutError
        except Exception as err:
            status = Status.EXCEPTION
            logger.error(ERR_INFO.format( err, initial_state, agents[0], agents[1], game_state, game_history ))
            traceback.print_exception(type(err), err, err.__traceback__)
            break
        finally:
            if time_limit and not debug:
                signal.signal(signal.SIGPROF, signal.SIG_IGN)      # Unregister the timeout signal

        if action not in game_state.actions():
            status = Status.INVALID_MOVE
            print(ERR_INFO.format( 'INVALID_MOVE', initial_state, agents[0], agents[1], game_state, game_history ))
            logger.error(ERR_INFO.format( 'INVALID_MOVE', initial_state, agents[0], agents[1], game_state, game_history ))
            break

        game_state = game_state.result(action)
        game_history.append(action)

        # Callbacks can be used to hook in additional functionality after each turn, such as verbose rendering
        callbacks = list(callbacks) if isinstance(callbacks, (tuple,list,set)) else [ callbacks ]
        if verbose: callbacks = [ verbose_callback ] + callbacks
        for callback in callbacks:
            if not callable(callback): continue
            callback(
                game_state=game_state,
                action=action,
                active_player=active_player,
                active_idx=active_idx,
                match_id=match_id
            )
    else:
        status = Status.GAME_OVER
        if game_state.utility(active_idx) > 0:
            winner, loser = loser, winner  # swap winner/loser if active player won

    logger.info(RESULT_INFO.format(status, game_state, game_history, winner, loser))
    return winner, game_history, match_id


def verbose_callback(game_state, action, active_player, active_idx, match_id):
    summary = "match: {} | {}({}) => {}".format(
        match_id,  active_player.__class__.__name__, active_idx, DebugState.ind2xy(action)
    )
    board = str(DebugState.from_state(game_state))
    print(summary); logger.info(summary)
    print(board);   logger.info(board)


def main():
    args         = argparser()
    test_agent   = TEST_AGENTS[args.opponent.upper()]
    custom_agent = Agent(CustomPlayer, "Custom Agent")
    players      = ( test_agent, custom_agent )
    results      = { player: 0 for player in players }
    match_count  = args.rounds * 2
    game_histories = []

    time_start = time.perf_counter()
    print("{} vs {} | Running {} games:".format(custom_agent.name, test_agent.name, match_count))
    for match_id in range(match_count):
        player_order = ( players[(match_id)%2], players[(match_id+1)%2] )  # reverse player order between matches
        winner, game_history, match_id = play_sync(player_order, match_id=match_id, **vars(args))
        results[winner] += 1
        game_histories.append(game_history)
        if not args.verbose:
            print('+' if winner == custom_agent else '-', end='', flush=True)

    time_taken = time.perf_counter() - time_start
    percentage = 100 * (results[custom_agent] / match_count)
    message    = "{} won {}/{} ({:.1f}%) of matches against {} in {:.0f}s ({:.2f}s/round)".format(
        custom_agent.name, results[custom_agent], match_count, percentage, test_agent.name, time_taken, time_taken/match_count
    )
    print()
    print(message); logger.info(message)
    print()

if __name__ == '__main__':
    main()
