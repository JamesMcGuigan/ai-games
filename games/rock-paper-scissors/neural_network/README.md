# Rock Paper Scissors - LSTM

This implements an LSTM for Rock Paper Scissors in Pytorch.

Kaggle Notebook: https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-lstm

# Install

The training loop can be executed via:
```
cd games/rock-paper-scissors/
pip3 install -r requirements.txt
./neutal_networks/rps_trainer.py
```

# NNBase Class

[NNBase.py](NNBase.py) is a generic baseclass to handle saving/loading the model from file and other utility functions

# LSTM Agent

This is the main neural network model class implemented in pytorch.

Input is encoded multiple ways:
```
x = torch.cat([
    noise.flatten(),    # tensor (3,)  random noise for probablistic moves
    step.flatten(),     # tensor (7,)  round(log(step)) one-hot encoded - to predict warmup periods    
    stats.flatten(),    # tensor (2,3) with move frequency percentages   
    history.flatten(),  # tensor (2,3,window=10) one-hot encoded timeseries history
])
```

Then fed through the following network:
```
RpsLSTM(
  (lstm): LSTM(76, 128, num_layers=3, batch_first=True, dropout=0.25)
  (dense_1): Linear(in_features=204, out_features=128, bias=True)
  (dense_2): Linear(in_features=128, out_features=128, bias=True)
  (out_probs): Linear(in_features=128, out_features=3, bias=True)
  (out_hash): Linear(in_features=128, out_features=128, bias=True)
  (activation): Softsign()
  (softmax): Softmax(dim=2)
)
```

It defines two custom loss functions:
- `loss_probs()` is for correctly predicting the next opponent move based on 1/0.5/0 EV scores
- `loss_hash()` is for predicting who our current opponent is

[Locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) is used within `loss_hash()`.
This is done by creating a xxhash(seed=0) of the agents str label,
then using modulo to place it into a fixed-size one-hot-encoded bucket.
Whist this information is not directly used for making moves,
the idea is to train the model to have an internal representation
of who our opponent is, such that it can better select an appropriate strategy

[Softsign](https://sefiks.com/2017/11/10/softsign-as-a-neural-networks-activation-function/)
`f(x) = x / (1 + |x|)` is used as the activation function. It has a similar shape to `tanh()`
but has two extra "bumps" in the curve which a neural network can make use of.
Its lesser known but has been cited in [recent papers](https://paperswithcode.com/method/softsign-activation).

I have not done a through exploration as to the best activation function to use here,
however it did fix a weird bug caused by [PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
that resulted in the LSTM weights converging to NaN after a thousand epochs.

I also concatinate the original input to the LSTM output before passing it the dense layers.
I am unsure if I should be doing this, or rely purely on this information being saved into the LSTM embedding.


# RPS Trainer

This is a training loop abstracted to play against arbitray kaggle_environment agents.

The RL interface for kaggle_environments is:
```
env         = make("rps", { "episodeSteps": steps }, debug=False)
trainer     = env.train(random.sample([None, agent], 2))  # random player order
observation = trainer.reset()
done        = False
while not done:
    action = model( observation.lastOpponentAction )
    observation, reward, done, info = trainer.step(action)
```

You can read more about this in the [docs](https://github.com/Kaggle/kaggle-environments)

In theory this training loop code could be easily repurposed for other kaggle competitions.

---

This training loop takes a dictionary of opponent agents, and simulates a 100 step = 50 round match against each one.
`loss.backward()` is only called at the end of each match as its the only way I have figured out
how to solve the `RuntimeError: Trying to backward through the graph a second time` exception.
If anybody knows a better way of doing this, please let me know.

I did have a bit of code to probablistically select agents based on their accuracy percentage.
The idea being to skip the rock/paper/scissors agents once they had reached 100% accuracy,
and focus most of the training time on the agents that needed the most training.
However when retraining from scratch it was causing weird effects in my output statitsics
(unsure if this was just a bug in my logging code). Its been disabled for now.

`RMSprop` seems to work better than other optimizers such as `Adam` or `Adadelta`
for this reinforcement learning task. I don't have any statistics on this,
so can't say if my observations where simply based on coincidence.

`CosineAnnealingWarmRestarts` was added in because several of the [DeepMind](https://deepmind.com/)
papers mention that they use a cosine based scheduling system for their models.
I am not 100% sure this is the same scheduler, or what the correct settings should be.
I am unsure if this is helping or hurting my model, and if I should leave it in or not.

For the sake of making it easier to read the directional trend of the logfile numbers,
I have implemented a very basic running average by taking the mean of the
current and next values in the sequence. This might not be the technically correct
way of doing it (its more like summing an infinite series), but gives  
approximately correct numbers and is a simple one-liner to smooth the curves
and make it easier to observe the directional trend of loss and accuracy.

Learning rate is currently set to `1e-4`. One observation from earlier in the development cycle
was that if the learning rate was set too high, such as `1e-1`, then the model would fail to train
even against the simplest Rock/Paper/Scissors agents. I am unsure if I have set it too low,
or how exactly this interacts with `CosineAnnealingWarmRestarts`.
Generally seems safer to be too small than too large.


A few technical points to remember:
- `model.train()` and `model.eval()` can be used to switch between training and production modes.
    - Dropout is disabled in production, and this may also have an effect on batch normalization layers.
- `model.load()` and `model.save()` are custom methods of my `NNBase` class
- `except KeyboardInterrupt: pass` ensures the model gets saved on Ctrl-C exit
- `if __name__ == '__main__':` prevents the training loop from running if we import `rps_trainer()` from a seperate file

---

Losses and accuracy are displayed based on running averages
```
   200 | losses = 0.419607 0.569539 | 100 r  99 p  98 s  99 seq  65 rotn  32 tree  35 iocaine  38 greenberg  47 stats
   210 | losses = 0.420502 0.617317 | 100 r  99 p  98 s  99 seq  70 rotn  30 tree  46 iocaine  40 greenberg  54 stats
   220 | losses = 0.411737 0.618371 | 100 r  99 p  98 s  99 seq  54 rotn  32 tree  45 iocaine  40 greenberg  51 stats
   230 | losses = 0.448281 0.667498 | 100 r  99 p  98 s  99 seq  52 rotn  27 tree  45 iocaine  42 greenberg  51 stats
```

The first number is the training epoch, which is a 100 step match against each agent

There are two loss functions:
0. loss_probs() - softmax probability vs EV score of opponent move prediction
1. loss_hash() - Categorical Cross Entropy loss for agent identity prediction using Locality-sensitive hashing

The last set of numbers are accuracy percentage scores in actual gameplay.
- 100 = 100% accuracy winning on every round
- 50  = 50% is a statistical draw, either via draw or win/loss every other round


# Opponents

We start with a series of really simple agents: Rock, Paper, Scissors, Sequential. These act as a baseline showing that the LSTM neural network is at least capable of learning simple logic.

Next up are a variety of more complex agents, implementing a range of different strategies
- [Anti-Rotn](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-anti-rotn)
- [Multi Stage Decision Tree](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-multi-stage-decision-tree)
- [Iocaine Powder](https://www.kaggle.com/jamesmcguigan/rps-roshambo-comp-iocaine-powder)
- [Greenberg](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-greenberg)
- [Statistical Prediction](https://www.kaggle.com/jamesmcguigan/rock-paper-scissors-statistical-prediction)


# Questions

This model can successfully defeat the simplest of agents such as:
- Rock
- Paper
- Scissors
- Sequential

It has problems however with more complex agents, where struggles to get beyond a draw
- anti_rotn
- multi_stage_decision_tree
- iocaine_powder
- greenberg

I am unsure exactly what I am doing wrong here.
- Is Rock Paper Scissors a suitable usecase for an LSTM network?
- Am I training for long enough. The model seemed to plateau at a draw for the advanced agents
- Are these more advanced agents too complex for a neural network to reverse engineer?
    - The simple agents are able to train with `hidden_size=16`
    - Do I need a much larger embedding size?
- Is my model too small or too large?
    - Are 3 LSTM layers better than 1
    - Should I have a pyramid of 3 dense layers rather than a square of 2
    - The Largest possible model that will fit in 100Mb is hidden_size=1024 with pyramid shaped dense layers, but this is very slow to train
- I concatenate the original input to the LSTM output before passing it the dense layers
    - Unsure if I should rely on the LSTM embedding to fully encode everything that needs to be remembered
- Am I missing any obvious layers such as batch normaliztion?
- Should I be training against a set of simpler agents to start with, such as: statistical, reactionary etc

Thank you for any advice or feedback on how to improve this work
