import random
import time
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List

from kaggle_environments import make

from memory.RPSNaiveBayes import naive_bayes_agent
from rng.random_agent import random_agent
from roshambo_competition.anti_rotn import anti_rotn
from roshambo_competition.greenberg import greenberg_agent
from roshambo_competition.iocaine_powder import iocaine_agent
from simple.anti_pi import anti_pi_agent
from simple.counter_reactionary import counter_reactionary
from simple.de_bruijn_sequence import de_bruijn_sequence
from simple.mirror import mirror_opponent_agent
from simple.paper import paper_agent
from simple.pi import pi_agent
from simple.reactionary import reactionary_agent
from simple.rock import rock_agent
from simple.scissors import scissors_agent
from simple.sequential import sequential_agent
from statistical.statistical import statistical
from statistical.statistical_prediction import statistical_prediction_agent
from tree.multi_stage_decision_tree import decision_tree_agent


class Simulations():
    agents = {
        'random':        random_agent,
        'rock':          rock_agent,
        'paper':         paper_agent,
        'scissors':      scissors_agent,
        'pi':            pi_agent,
        'anti_pi':       anti_pi_agent,
        'de_bruijn':     de_bruijn_sequence,
        'sequential':    sequential_agent,
        'reactionary':   reactionary_agent,
        'counter_react': counter_reactionary,
        'mirror':        mirror_opponent_agent,
        'statistical':   statistical,
        'anti_rotn':     anti_rotn,
        'tree':          decision_tree_agent,
        'iocaine':       iocaine_agent,
        'greenberg':     greenberg_agent,
        'stat_pred':     statistical_prediction_agent,
        'naive_bayes':   naive_bayes_agent,
    }

    def __init__(self, decay=0.95, confidence=0.33, first_action=1, verbose=True):
        self.verbose      = bool(verbose)
        self.decay        = float(decay)
        self.confidence   = float(confidence)
        self.first_action = int(first_action)

        self.step    = 0
        self.history = {
            "opponent": [],
            "action":   [],
        }
        self.envs = {
            agent_name: make("rps", debug=False)
            for agent_name in self.agents.keys()
        }
        self.trainers = {
            agent_name: self.envs[agent_name].train([None, agent])
            for agent_name, agent in self.agents.items()
        }
        self.observations = {
            agent_name: trainer.reset()
            for agent_name, trainer in self.trainers.items()
        }
        self.predictions = {
            agent_name: []
            for agent_name in self.agents.keys()
        }
        self.done = {
            agent_name: False
            for agent_name in self.agents.keys()
        }
        self.timer = defaultdict(float)

    def __call__(self, obs, conf):
        return self.agent(obs, conf)

    # obs  {'remainingOverageTime': 60, 'step': 1, 'reward': 0, 'lastOpponentAction': 0}
    # conf {'episodeSteps': 10, 'actTimeout': 1, 'runTimeout': 1200, 'signs': 3, 'tieRewardThreshold': 20, 'agentTimeout': 60}
    def agent(self, obs, conf):
        time_start = time.perf_counter()
        self.update_state(obs, conf)
        self.update_trainers()

        if obs.step == 0:
            best_agent = 'first_action'
            action     = self.first_action % conf.signs
            expected   = (action - 1)      % conf.signs
        else:
            best_agent = self.pick_best_agent()
            expected   = self.predictions[best_agent][0]
            action     = int(expected + 1) % conf.signs

        self.history['action'].insert(0, action)
        if self.verbose:
            best_score = self.prediction_best_score()
            time_taken = time.perf_counter() - time_start
            print(f'time = {time_taken:0.2f} | step = {obs.step:4d} | action = {expected} -> {action} | {best_score:.3f} {best_agent}')
        return int(action) % conf.signs



    ### State Management

    def update_state(self, obs, conf):
        self.step = obs.step
        if obs.step > 0:
            self.history['opponent'].insert(0, obs.lastOpponentAction % conf.signs)


    def update_trainers(self):
        last_action = (self.history['action'] or [self.first_action])[0]
        for agent_name, trainer in self.trainers.items():
            if not self.done[agent_name]:
                time_start = time.perf_counter()
                observation, reward, done, info = trainer.step(last_action)
                time_taken = time.perf_counter() - time_start

                self.timer[agent_name]        = time_taken
                self.observations[agent_name] = observation
                self.done[agent_name]         = done
                self.predictions[agent_name].insert(0, observation.lastOpponentAction)



    ### Scoring and Agent Selection

    def pick_best_agent(self) -> str:
        # If we are not confident in our prediction, default to random
        best_score = self.prediction_best_score()
        if not best_score or best_score < self.confidence:
            best_agent = 'random'
        else:
            scores = self.prediction_scores()
            best_agents = [
                agent_name
                for agent_name, agent_score in scores.items()
                if agent_score == best_score
            ]
            best_agent = random.choice(best_agents)
        return best_agent


    def prediction_best_score(self) -> float:
        scores     = self.prediction_scores()
        best_score = max(scores.values()) if len(scores) else 0.0
        return best_score

    def prediction_scores(self) -> Dict[str, float]:
        scores = {
            agent_name: self.prediction_score(predictions, agent_name)
            for agent_name, predictions in self.predictions.items()
            if not self.done[agent_name]
        }
        scores = dict(sorted(scores.items(), key=itemgetter(1), reverse=True))
        return scores

    def prediction_score(self, predictions: List[int], agent_name=None) -> float:
        # NOTE: step == 1 | predictions == [1,0] | self.history['opponent'] == [0]
        rewards = []
        total   = 0.0
        for n, prediction in enumerate(predictions[1:]):
            actual = self.history['opponent'][n]
            reward = self.reward_prediction(prediction, actual) * (self.decay ** n)
            total += (self.decay ** n)
            rewards.append(reward)
        return sum(rewards) / total if len(rewards) else 0.0

    def reward_prediction(self, prediction: int, actual: int) -> float:
        action   = (prediction + 1) % 3
        opponent = actual
        if (action - 1) % 3 == opponent % 3: return  1.0  # win
        if (action - 0) % 3 == opponent % 3: return  0.5  # draw
        if (action + 1) % 3 == opponent % 3: return  0.0  # loss
        return 0.0


simulations_instance = Simulations()
def simulations_agent(obs, conf):
    return simulations_instance.agent(obs, conf)
