# Rock Paper Scissors - LSTM

RpsLSTM.py implements an LSTM agent for Rock Paper Scissors

The training loop can be executed via:
```
cd games/rock-paper-scissors/
pip3 install -r requirements.txt
./neutal_networks/rps_trainer.py
```

Model is saved to: `neural_network/models/RpsLSTM.pth`

The first number is the training epoch, which is a 100 step match against each agent

There are two loss functions:
0. loss_probs() - softmax probability vs EV score of opponent move prediction
1. loss_hash() - Categorical Cross Entropy loss for agent identity prediction using Locality-sensitive hashing

The last set of numbers are accuracy percentage scores in actual gameplay. 
- 100 = 100% accuracy winning on every round
- 50  = 50% is a statistical draw, either via draw or win/loss every other round

Losses and accuracy are displayed based on running averages
```
   200 | losses = 0.419607 0.569539 | 100 r  99 p  98 s  99 seq  65 rotn  32 tree  35 iocaine  38 greenberg  47 stats
   210 | losses = 0.420502 0.617317 | 100 r  99 p  98 s  99 seq  70 rotn  30 tree  46 iocaine  40 greenberg  54 stats
   220 | losses = 0.411737 0.618371 | 100 r  99 p  98 s  99 seq  54 rotn  32 tree  45 iocaine  40 greenberg  51 stats
   230 | losses = 0.448281 0.667498 | 100 r  99 p  98 s  99 seq  52 rotn  27 tree  45 iocaine  42 greenberg  51 stats
```


Using the parameters
```
model = RpsLSTM(hidden_size=128, num_layers=3, dropout=0.25).train()
rps_trainer(model, agents, steps=100, lr=1e-4)
```

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
