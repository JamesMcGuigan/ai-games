# Game of Life Forward - Hardcoded and Trained Miniature Neural Networks

This is a response to the paper: 

**Why neural networks struggle with the Game of Life**
by Jacob M. Springer, Garrett T. Kenyon
- https://arxiv.org/abs/2009.01398 

> Efforts to improve the learning abilities of neural networks have focused mostly 
> on the role of optimization methods rather than on weight initializations. 
> Recent findings, however, suggest that neural networks rely on lucky random initial weights of subnetworks 
> called "lottery tickets" that converge quickly to a solution. 

> To investigate how weight initializations affect performance, we examine small convolutional networks 
> that are trained to predict n steps of the two-dimensional cellular automaton Conway's Game of Life, 
> the update rules of which can be implemented efficiently in a 2n+1 layer convolutional network. 

> We find that networks of this architecture trained on this task rarely converge. 
> Rather, networks require substantially more parameters to consistently converge. 
> In addition, near-minimal architectures are sensitive to tiny changes in parameters: 
> changing the sign of a single weight can cause the network to fail to learn. 

> Finally, we observe a critical value d_0 such that training minimal networks with examples 
> in which cells are alive with probability d_0 dramatically increases the chance of convergence to a solution. 
> We conclude that training convolutional neural networks to learn the input/output function represented 
> by n steps of Game of Life exhibits many characteristics predicted by the lottery ticket hypothesis, namely, 
> that the size of the networks required to learn this function are often significantly larger than the minimal network 
> required to implement the function.

Which was further discussed in the blog post:
- https://bdtechtalks.com/2020/09/16/deep-learning-game-of-life/


# First Attempt - GameOfLifeForward_128

The first attempt at getting a neural network to train to 100% accuracy was the 
[GameOfLifeForward_128](./GameOfLifeForward_128.py) model. Which was documented in this Kaggle Notebook 
- https://www.kaggle.com/jamesmcguigan/pytorch-game-of-life

This model, with 128 CNN layers, is greatly oversized compared to the theoretical minimum, 
but it reliably trains from random weight initialization acted as proof of concept that a 
neural network can indeed be trained to 100% accuracy.


# Manually Hardcoded Weights
- [GameOfLifeHardcodedLeakyReLU.py](./GameOfLifeHardcodedLeakyReLU.py)
- [GameOfLifeHardcodedReLU1_21.py](./GameOfLifeHardcodedReLU1_21.py)
- [GameOfLifeHardcodedReLU1_41.py](./GameOfLifeHardcodedReLU1_41.py)
- [GameOfLifeHardcodedTanh.py](./GameOfLifeHardcodedTanh.py)

Inspired by the above paper, I decided to create a minimalist neural network with hardcoded weights.

The boolean rules of the Game of Life requires counting the number of neighbouring cells,
comparing it to the value of the center cell, and then performing a less than or greater than operation.

The rules can be expressed one of two ways:
```
AND(
    z3.AtLeast( past_cell, *past_neighbours, 3 ): n >= 3
    z3.AtMost(             *past_neighbours, 3 ): n <  4
)
```
```
SUM(
    Alive && neighbours >= 2
    Alive && neighbours <= 3
    Dead  && neighbours >= 3
    Dead  && neighbours <= 3
) >= 2
```

# AtLeast/AtMost ruleset
- Source: [GameOfLifeHardcodedReLU1_21.py](./GameOfLifeHardcodedReLU1_21.py)
- Source: [GameOfLifeHardcodedTanh.py](GameOfLifeHardcodedTanh.py)

## Counting Neighbours

The original value of the board can either be solved via passthrough of the original board state,
or implemented below as 3x3 convolution layer with only the center weight set to 1.0:
```
x = torch.cat([ x, input ], dim=1)   
```

The summing of neighbours (and passthrough of the original cell) can be expressed using 3x3 convolutions.

The input data is binary 1 or 0, so the convolution will output +1 for every Alive cell found 
in the specified position. The result is simply a count of neighbours, which is conceptually similar
to [AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) \* 9, but
without downsampling the board size. 
```
self.counter.weight.data = torch.tensor([
    [[[ 0.0, 0.0, 0.0 ],
      [ 0.0, 1.0, 0.0 ],
      [ 0.0, 0.0, 0.0 ]]],

    [[[ 1.0, 1.0, 1.0 ],
      [ 1.0, 0.0, 1.0 ],
      [ 1.0, 1.0, 1.0 ]]]
])
```

## GreaterThan (>) and LessThan (<)

GreaterThan (>) and LessThan (<) and  can be expressed with the combination of ReLU1 plus a bias.
```
self.logics[0].weight.data = torch.tensor([
    [[[ 1.0 ]], [[  1.0 ]]],   # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
    [[[ 0.0 ]], [[ -1.0 ]]],   # n <  4   # z3.AtMost(             *past_neighbours, 3 ),
])
self.logics[0].bias.data = torch.tensor([
    -3.0 + 1.0,               # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
    +3.0 + 1.0,               # n <= 3   # z3.AtMost(             *past_neighbours, 3 ),
])
ReLU1 = nn.ReLU6()(x * 6.0) / 6.0
```

If the cell is alive, and has 5 neighbours, then 
- logics[0][0] = (1.0 * Alive + 1.0 * 5 Neighbours) + (-3.0 + 1.0) bias = +4.0  # positive if n >= 3
- logics[0][1] = (0.0 * Alive - 1.0 * 5 Neighbours) + (+3.0 + 1.0) bias = -1.0  # positive if n <= 3

If the cell is dead, and has 3 neighbours, then 
- logics[0][0] = (0.0 * Alive + 1.0 * 3 Neighbours) - (-3.0 + 1.0) bias = +1.0  # positive if n >= 3
- logics[0][1] = (0.0 * Alive - 1.0 * 3 Neighbours) + (+3.0 + 1.0) bias = +1.0  # positive if n <= 3 

A +-3.0 bias will convert a neighbours count of 3 to 0.0, so we need to add +1.0 in the direction
of the bias to produce a positive > +1.0 output when the condition is satisfied. 


## ReLU1

Now that we have encoded the AtLeast/AtMost conditions as positive or negative numbers,
we need a logical AND gate to assert that both conditions are true.

A standard ReLU will set any negative values to 0.0, but returns the distance
away from the bias point in the positive direction.  
This V shaped activation does not allow for a simple implementation of a logical AND gate.

For binary logic, a ReLU1 activation `min(max(0,n),1)` is Z shaped (like Sigmoid and Tanh) 
and outputs in the domain `[0,1]` discarding any information about distances greater than 1.0 
away from the bias point. When working with binary data, this effectively implements cast to boolean.


## AND Logic Gate

Given the presence of ReLU1, the logic for the AND gate is simple.
- Both of the input statements need to be true. 
- ReLU1 enforces that the inputs will be in the domain `[0,1]`.
- The maximum sum from the weights is 2.0
- A bias of -1.0 will return 1.0 if both are True, or `<= 0` for False
- ReLU1 activation on the output will conver this to either 1.0 or 0.0

```
self.output.weight.data = torch.tensor([[
    [[  1.0 ]],
    [[  1.0 ]],
]])
self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # sum() >= 2
```

The [GameOfLifeHardcodedTanh.py](GameOfLifeHardcodedTanh.py) implementation follows similar logic 
but with numbers suited to Tanh() which is a continuous function which outputs in the domain `(-1,1)`


# AND/OR Implementation 
- Source: [GameOfLifeHardcodedReLU1_41.py](./GameOfLifeHardcodedReLU1_41.py)

The ruleset is classically expressed using boolean logic:
```
Alive && ( neighbours >= 2 && neighbours <= 3 ) ||
Dead  && ( neighbours >= 3 && neighbours <= 3 )
```

OR(AND, AND) requires three layers to properly express, however this logic can be
equivalently expressed in only two layers using SUM(AND):
```
SUM(
    Alive && neighbours >= 2
    Alive && neighbours <= 3
    Dead  && neighbours >= 3
    Dead  && neighbours <= 3
) >= 2
```

This requires a slightly larger network with 4 channels, one for each condition.  


## Counter

The counter weights remain the same:
```
self.counter.weight.data = torch.tensor([
    [[[ 0.0, 0.0, 0.0 ],
      [ 0.0, 1.0, 0.0 ],
      [ 0.0, 0.0, 0.0 ]]],

    [[[ 1.0, 1.0, 1.0 ],
      [ 1.0, 0.0, 1.0 ],
      [ 1.0, 1.0, 1.0 ]]]
])
```

## AND( bool, > )

The `Alive|Dead &&` logic can be encoded in the same layer as the GreaterThan (>) and LessThan (<) logic,
by using different orders of magnitudes for the weights.

The output domain for the 3x3 counter convolution is `[0, 8]`.

If we set a weight of +-10 for the center cell, then this value is high enough to 
completely offset the maximum value is 8 from the neighbours count. 
This allows us to implement `AND( bool, > )` within a single layer.

```
self.logics[0].weight.data = torch.tensor([
    [[[  10.0 ]], [[  1.0 ]]],  # Alive && neighbours >= 2
    [[[  10.0 ]], [[ -1.0 ]]],  # Alive && neighbours <= 3
    [[[ -10.0 ]], [[  1.0 ]]],  # Dead  && neighbours >= 3
    [[[ -10.0 ]], [[ -1.0 ]]],  # Dead  && neighbours <= 3
])
self.logics[0].bias.data = torch.tensor([
    -10.0 - 2.0 + 1.0,  # Alive +  neighbours >= 2
    -10.0 + 3.0 + 1.0,  # Alive + -neighbours <= 3
      0.0 - 3.0 + 1.0,  # Dead  +  neighbours >= 3
      0.0 + 3.0 - 1.0,  # Dead  + -neighbours <= 3
])
```

If the cell is alive, and has 5 neighbours, then 
- logics[0][0] = ( 10.0 * Alive + 1.0 * 5 Neighbours) + (-10.0 - 2.0 + 1.0) bias = +15.0 - 11.0 == +4.0
- logics[0][1] = ( 10.0 * Alive - 1.0 * 5 Neighbours) + (-10.0 + 3.0 + 1.0) bias =  +5.0 -  6.0 == -1.0
- logics[0][2] = (-10.0 * Alive + 1.0 * 5 Neighbours) + (  0.0 - 3.0 + 1.0) bias =  -5.0 -  2.0 == -7.0
- logics[0][3] = (-10.0 * Alive - 1.0 * 5 Neighbours) + (  0.0 + 3.0 + 1.0) bias =  -5.0 +  4.0 == -1.0


If the cell is dead, and has 3 neighbours, then 
- logics[0][0] = ( 10.0 * Dead + 1.0 * 3 Neighbours) + (-10.0 - 2.0 + 1.0) bias =  +3.0 - 11.0 ==  -6.0
- logics[0][1] = ( 10.0 * Dead - 1.0 * 3 Neighbours) + (-10.0 + 3.0 + 1.0) bias =  -3.0 -  6.0 == -11.0
- logics[0][2] = (-10.0 * Dead + 1.0 * 3 Neighbours) + (  0.0 - 3.0 + 1.0) bias =  +3.0 -  2.0 ==  +3.0
- logics[0][3] = (-10.0 * Dead - 1.0 * 3 Neighbours) + (  0.0 + 3.0 - 1.0) bias =  -3.0 +  4.0 ==  +1.0

The top two expressions can only be True/Positive if we have a +10 contribution from the Alive cell.
The bottom two expressions will automatically be False/Negative when Alive == -10

ReLU1 then converts to: Positive = 1.0 and Negative = 1.0


## SUM( AND )

Both of the Alive or Dead statements (2 clauses each) needs to be True.

This can be implemented as  `sum() >= 2`, given the following preconditions:
- The -10 weight above makes the groups of clauses mutually exclusive
- Each clause group has the same size 
    - for imbalanced group sizes, clauses could be duplicated to make the group sizes the same

Without these preconditions, a second layer would be required to formally implement:
- OR( AND(input[0], input[1]), AND(input[3], input[4]) )

```
self.output.weight.data = torch.tensor([[
    [[  1.0 ]],
    [[  1.0 ]],

    [[  1.0 ]],
    [[  1.0 ]],
]])
self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # sum() >= 2
```


# Training from Randomized Weights

Having manually found solutions for a minimalist neural network architecture, 
the question is if such a network can be trained using gradient decent starting from random weight initialization?

The choice of activation function has great significance here.

Using ReLU1 or Tanh, resulted in the network only being able to solve the final `self.output` layer, 
and required the weights for the `self.logics[0]` and `self.counter` layers to be hardcoded.

LeakyReLU was able to solve the `self.output` and `self.logics[0]` layers given hardcoded `self.counter` weights.

PReLU is a version of LeakyReLU, but with a parametrized weight for the negative slope. 
This feature greatly improves its ability to solve this problem and 
can (usually) solve the entire network including the weights for the 3x3 convolutional `self.counter`.

A variety of minimalist architectures where tested to discover the smallest network capable of reliably 
solving the Game of Life forward play problem, and the frequency of which "lottery ticket" initialization 
is required.



## Minimalist Network Architectures

Network architectures where chosen with 1, 2 or 4 convolutional 3x3 layers, 
both with and without passthrough of the original board state after the convolutional layer.

In hindsight it may have been better to use ReLU1 as the final activation function rather than sigmoid 
as we are outputting binary values.

[GameOfLifeForward_1.py](GameOfLifeForward_1.py) - 1 * 3x3 (with passthrough)
```
Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
x = torch.cat([ x, input ], dim=1)
Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
nn.sigmoid()
```

[GameOfLifeForward_1N.py](GameOfLifeForward_1N.py) - 1 * 3x3 (without passthrough)
```
Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
nn.sigmoid()
```

[GameOfLifeForward_2.py](GameOfLifeForward_2.py) - 2 * 3x3 (with passthrough)
```
Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
x = torch.cat([ x, input ], dim=1)
Conv2d(3, 2, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
nn.sigmoid() 
```

[GameOfLifeForward_2N.py](GameOfLifeForward_2N.py) - 2 * 3x3 (without passthrough)
```
Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
nn.sigmoid() 
```

[GameOfLifeForward_4.py](GameOfLifeForward_4.py) - 2 * 3x3 (with passthrough)
```
Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
Conv2d(5, 4, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(4, 1, kernel_size=(1, 1), stride=(1, 1))
nn.sigmoid() 
```

[GameOfLifeForward_128.py](GameOfLifeForward_128.py) - original implementation
```
Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
nn.PReLU()
Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
nn.PReLU()
Conv2d(9, 1, kernel_size=(1, 1), stride=(1, 1))
nn.sigmoid() 
```

## Training
- Source: [logs/GameOfLifeForward.sh](./logs/GameOfLifeForward.sh)
- Source: [logs/GameOfLifeForward.stats.sh](./logs/GameOfLifeForward.stats.sh)

Each network was trained a total of 10 times, with weights reinitialized each time using `nn.init.kaiming_normal_()`

Training was continued until it had correctly predicted 100,000 boards in a row to 100% accuracy.

A 20 minute timeout was introduced after observing that the networks would usually converge within 7 minutes,
or else get stuck in a local minimum and never converge even after an hour of runtime. 
The maximum observed runtime (after timeout was introduced) for a convergent result was 13 minutes (about 2x average runtime). 

Dropout was disabled for GameOfLifeForward_128 as it caused excessive (hour+) training times 
as it tried to build in sufficient redundancy into the network to transition from 99.999% accuracy 
to 100% accuracy.

Results are as follows:
```

GameOfLifeForward_1   |  9/10 successes | epochs = 2254 / 5313 / 10318 | time seconds = 183 / 420 / 807 (min/avg/max)
GameOfLifeForward_1N  |  6/10 successes | epochs = 3376 / 4771 /  7097 | time seconds = 262 / 370 / 552 (min/avg/max)
GameOfLifeForward_2   |  9/10 successes | epochs = 3047 / 4878 /  8587 | time seconds = 239 / 379 / 669 (min/avg/max)
GameOfLifeForward_2N  |  8/10 successes | epochs = 3245 / 4604 /  6924 | time seconds = 252 / 356 / 533 (min/avg/max)
GameOfLifeForward_4   | 10/10 successes | epochs = 2027 / 3308 /  5061 | time seconds = 160 / 263 / 420 (min/avg/max)
GameOfLifeForward_128 | 10/10 successes | epochs =  441 /  508 /   570 | time seconds =  52 /  80 /  93 (min/avg/max)
```

As we can see larger network sizes generally train faster are less subject to lottery ticket initialization.
Passthrough of the original board state reduces the frequency of "lottery ticket" weight initialization fails.

What is most interesting is that gradient decent was able to solve the GameOfLifeForward_1N architecture 
with only a single 3x3 convolution and no passthrough 60% of the original state.

Here is an example of one of the solutions it discovered:
```
GameOfLifeForward_1(
  (criterion): MSELoss()
  (layers): ModuleList(
    (0): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode=circular)
    (1): Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
    (2): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
  )
  (dropout): Dropout(p=0.0, inplace=False)
  (activation): PReLU(num_parameters=1)
)
activation.weight
[0.25]

layers.0.weight
[[[[ 0.26990238 -0.20259705 -0.4364899 ]
   [ 0.55158144  0.25551328  0.3859689 ]
   [-0.24950361  0.2531228  -0.74614656]]]]

layers.0.bias
[0.1]

layers.1.weight
[[[[-1.3824693 ]] [[ 0.47230604]]]
 [[[ 0.85298234]] [[ 1.6202106 ]]]]

layers.1.bias
[0.1 0.1]

layers.2.weight
[[[[-0.11759763]] [[ 1.0376352 ]]]]

layers.2.bias
[0.1]
```


# Unit Tests
- Source: [neural_networks/tests/test_GameOfLifeHardcoded.py](../../neural_networks/tests/test_GameOfLifeHardcoded.py)

To validate that these solutions are indeed correct and reliable, 
the following unit test successfully asserts 100% correct results 
over a sequence of 10,000 randomly generated boards. 

```
models = [
    GameOfLifeHardcodedLeakyReLU(),
    GameOfLifeHardcodedReLU1_41(),
    GameOfLifeHardcodedReLU1_21(),
    GameOfLifeHardcodedTanh(),

    GameOfLifeForward_128(),
    GameOfLifeForward_2N(),
    GameOfLifeForward_1N(),
    GameOfLifeForward_4(),
    GameOfLifeForward_2(),
    GameOfLifeForward_1(),
]
@pytest.mark.parametrize("model", models)
def test_GameOfLifeHardcoded_generated_boards(model):
    inputs   = generate_random_boards(10_000)
    expected = life_steps(boards_t0)
    outputs  = model.predict(inputs)
    assert np.array_equal( outputs, expected )  # assert 100% accuracy
```
