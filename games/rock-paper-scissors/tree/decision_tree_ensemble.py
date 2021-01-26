# %%writefile submission.py
# Source: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-decision-tree-ensemble

import random
from collections import defaultdict
from operator import itemgetter

import numpy as np
from pydash import get, uniq
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


class RPSDecisionTreeEnsemble:
    def __init__(
            self,
            min_window   = 4,
            max_window   = 10,
            max_history  = 500,
            cutoff       = 0.33,
            warmup       = 5,
            model        = "tree",
            model_config = {
                # "criterion":             "entropy",
                # "ccp_alpha":             0.05,
                # "min_impurity_decrease": 0.1,
                # "min_samples_leaf":      0.1
            },
            verbose=True
    ):
        self.obs  = None
        self.conf = None

        self.min_window   = min_window
        self.max_window   = max_window
        self.max_history  = max_history
        self.cutoff       = cutoff
        self.warmup       = warmup
        self.model        = model
        self.model_config = model_config
        self.verbose      = verbose

        self.history = {
            "step":        [],
            "reward":      [],
            "action":      [],
            "opponent":    [],
            "rotn_self":   [],
            "rotn_opp":    [],
            "predictions": defaultdict(list),
        }
        # self.winrates = {}
        pass

    def __len__(self):
        lengths = [
            len(value)
            for key, value in self.history.items()
            if isinstance(value, list)
        ]
        return min(lengths) if len(lengths) else 0


    def __call__(self, obs, conf):
        return self.agent(obs, conf)


    # obs  {'remainingOverageTime': 60, 'step': 1, 'reward': 0, 'lastOpponentAction': 0}
    # conf {'episodeSteps': 1000, 'actTimeout': 1, 'runTimeout': 1200, 'signs': 3, 'tieRewardThreshold': 20, 'agentTimeout': 60}
    def agent(self, obs, conf):
        self.update_state(obs, conf)

        expected    = self.random_action()
        predictions = []
        if obs.step >= self.warmup:
            predictions = [
                ( expected, prob, model, window, target, source )
                for window in range(self.min_window, min(self.max_window, self.warmup))
                # for target in [ "action", "opponent", "rotn_self", "rotn_opp" ]
                for target in [ "opponent" ]
                for source in [
                    [ "action",    "opponent" ],
                    [ "rotn_self", "rotn_opp" ],
                    [ "step", "reward", "action", "opponent", "rotn_self", "rotn_opp" ]
                ]
                for expected, prob, model in [
                    self.predict(target=target, source=source, window=window)
                ]
                if prob > self.cutoff
            ]
            predictions = sorted(predictions, key=itemgetter(1), reverse=True)
            if len(predictions):
                expected, prob, model, window, target, source = predictions[0]

            # for key, expected, prob in predictions:
            #    self.history['predictions'][key].insert(0, expected)

        action = int(expected + 1) % conf.signs
        action = int(action or 0)  % conf.signs
        self.history['action'].insert(0, action)

        if self.verbose:
            print('expected', expected)
            print('action', action)
            print('self.history', self.history)
            if len(predictions):
                print('prediction')
                for expected, prob, model, window, target, source in predictions:
                    print(expected, prob, model, window, target, source)
                    self.plot(model, target, source)
            print()
        return action


    def random_action(self):
        return random.randint(0, self.conf.signs-1)


    def update_state(self, obs, conf):
        # Front load data, so self.history[0] is latest entry
        self.obs  = obs
        self.conf = conf
        self.history['step'].insert(0, obs.step )

        if obs.step != 0:
            last_reward = obs.reward - get(self.history['reward'], -1, 0)
            self.history['reward'].insert(0, last_reward)
            self.history['opponent'].insert(0, obs.lastOpponentAction )

        if len(self.history['opponent']) >= 1 and len(self.history['action']) >= 1:
            rotn_opp = (self.history['opponent'][0] - self.history['action'][0]  ) % conf.signs
            self.history['rotn_opp'].insert(0, rotn_opp )

        if len(self.history['opponent']) >= 2:
            rotn_self = (self.history['opponent'][0] - self.history['opponent'][1]) % conf.signs
            self.history['rotn_self'].insert(0, rotn_self )


    def predict(self, target='opponent', source=[], window=6, model_config={}):
        source  = uniq(source)
        fields  = uniq([ target ] + source)
        dataset = {
            name: get(self.history, name)
            for name in fields
        }
        max_size = min(map(len, dataset.values()))
        max_size = min(max_size, self.max_history)

        expected = self.random_action()
        prob     = 0.0
        model    = None
        try:
            if max_size > window:
                X = np.stack([
                    np.array([
                        get(self.history, name)[n:n+window]
                        for name in fields
                    ]).flatten()
                    for n in range(max_size-window)
                ])
                Y = np.array([
                    get(self.history, target)[n+window]
                    for n in range(max_size-window)
                ])
                Z = np.array([
                    get(self.history, name)[0:0+window]
                    for name in fields
                ]).flatten().reshape(1, -1)

                model = self.get_model(model_config)
                model.fit(X, Y)
                expected = model.predict(Z)[0]
                index    = model.classes_.tolist().index(expected)
                probs    = model.predict_proba(Z)
                prob     = probs[0][index]
        except Exception as exception:
            raise exception
            pass
        return expected, prob, model


    def get_model(self, model_config):
        model_config = { **self.model_config, **model_config }
        # DOCS: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        if self.model == "tree":
            return DecisionTreeClassifier( **model_config )
        assert self.model in ["tree"]


    # DOCS: https://mljar.com/blog/visualize-decision-tree/
    # DOCS: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_text.html
    def plot(self, model, target, source):
        if model is None: return
        if self.verbose:
            fields = uniq( [ target ] + source )
            # print( f'target = {target} | source = {source}' )
            print(tree.export_text(
                model,
                 # feature_names=fields,
                 show_weights=True
            ))


instance = RPSDecisionTreeEnsemble()
def kaggle_agent(obs, conf):
    return instance.agent(obs, conf)
