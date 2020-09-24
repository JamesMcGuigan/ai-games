#!/usr/bin/env python3
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from neural_networks.device import device
from neural_networks.GameOfLifeForward import gameOfLifeForward
from utils.game import generate_random_board
from utils.game import life_step


def train(model, batch_size=25, l1=0, l2=0, timeout=0):
    print(f'Training: {model.__class__.__name__}')
    time_start = time.perf_counter()

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)


    # epoch: 14481 | board_count: 362000 | loss: 0.0000385726 | accuracy = 0.9990336000 | time: 0.965ms/board
    scheduler = None

    # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board
    # Finished Training: GameOfLifeForward - 240995 epochs in 3569.1s
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        max_lr=1,
        base_lr=1e-3,
        step_size_up=250,
        mode='exp_range',
        gamma=0.8
    )

    try:
        model.load()
        num_params = torch.sum(torch.tensor([
            torch.prod(torch.tensor(param.shape))
            for param in model.parameters()
        ]))

        epoch        = 0
        board_count  = 0
        last_loss    = np.inf
        loop_loss    = 0
        loop_acc     = 0
        loop_count   = 0
        epoch_losses     = [last_loss]
        epoch_accuracies = [ 0 ]
        for epoch in range(1, sys.maxsize):
            if np.min(epoch_accuracies[-10000//batch_size:]) == 1.0:    break  # multiple epochs of 100% accuracy to pass
            if timeout and timeout < time.perf_counter() - time_start:  break
            epoch      += 1
            epoch_start = time.perf_counter()

            dataset  = [ generate_random_board() for _ in range(batch_size) ]
            expected = [ life_step(board)        for board in dataset ]
            inputs   = model.cast_inputs([ board for board in dataset  ]).to(device)
            labels   = model.cast_inputs([ board for board in expected ]).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            l1_loss = torch.sum(torch.tensor([ torch.sum(torch.abs(param)) for param in model.parameters() ])) # / num_params
            l2_loss = torch.sum(torch.tensor([ torch.sum(param**2)         for param in model.parameters() ])) # / num_params
            loss   += ( l1_loss * l1 ) + ( l2_loss * l2 )
            # loss   *= ( l1_loss * 1 )
            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step(loss)

            last_loss     = loss.item() / batch_size
            last_accuracy = torch.sum( outputs.to(torch.bool) == labels.to(torch.bool) ).cpu().numpy() / np.prod(outputs.shape)


            epoch_losses.append(last_loss)
            epoch_accuracies.append( last_accuracy )

            loop_loss   += last_loss
            loop_acc    += last_accuracy
            loop_count  += 1
            board_count += batch_size
            epoch_time   = time.perf_counter() - epoch_start

            # Print statistics after each epoch
            if board_count % 1000 == 0:
                print(f'epoch: {epoch:4d} | board_count: {board_count:5d} | loss: {loop_loss/loop_count:.10f} | accuracy = {loop_acc/loop_count:.10f} | time: {1000*epoch_time/batch_size:.3f}ms/board')
                loop_loss  = 0
                loop_acc   = 0
                loop_count = 0

            # print("epoch_losses[-10:]", epoch_losses[-10:], ' min = ', np.min(epoch_losses[-10:]))
            # model.save()


    except KeyboardInterrupt:
        pass
    except Exception as exception:
        print(exception)
        raise exception
    finally:
        time_taken = time.perf_counter() - time_start
        print(f'Finished Training: {model.__class__.__name__} - {epoch} epochs in {time_taken:.1f}s')
        model.save()


if __name__ == '__main__':
    train(gameOfLifeForward)
