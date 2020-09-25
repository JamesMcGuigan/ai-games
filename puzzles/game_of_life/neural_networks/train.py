#!/usr/bin/env python3
import atexit
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch import tensor

from neural_networks.device import device
from utils.game import generate_random_board
from utils.game import life_step


def train(model, batch_size=25, l1=0, l2=0, timeout=0, reverse_input_output=False):
    print(f'Training: {model.__class__.__name__}')
    time_start = time.perf_counter()

    atexit.register(model.save)      # save on exit - BrokenPipeError doesn't trigger finally:
    model.load().train().unfreeze()  # enable training and dropout

    # NOTE: criterion loss function now defined via model.loss()
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
    try:
        for epoch in range(1, sys.maxsize):
            if np.min(epoch_accuracies[-10000//batch_size:]) == 1.0:    break  # multiple epochs of 100% accuracy to pass
            if timeout and timeout < time.perf_counter() - time_start:  break
            epoch_start = time.perf_counter()

            inputs_np   = [ generate_random_board() for _     in range(batch_size) ]
            expected_np = [ life_step(board)        for board in inputs_np         ]
            inputs      = model.cast_inputs(inputs_np).to(device)
            expected    = model.cast_inputs(expected_np).to(device)

            # This is for GameOfLifeReverseOneStep() function, where we are trying to learn the reverse function
            if reverse_input_output:
                inputs_np, expected_np = expected_np, inputs_np
                inputs,    expected    = expected,    inputs
                assert np.all( life_step(expected_np[0]) == inputs_np[0] )


            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = model.loss(outputs, expected, inputs)
            if l1 or l2:
                l1_loss = torch.sum(tensor([ torch.sum(torch.abs(param)) for param in model.parameters() ])) / num_params
                l2_loss = torch.sum(tensor([ torch.sum(param**2)         for param in model.parameters() ])) / num_params
                loss   += ( l1_loss * l1 ) + ( l2_loss * l2 )

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                # scheduler.step(loss)  # only required for
                scheduler.step()

            # noinspection PyTypeChecker
            last_accuracy = model.accuracy(outputs, expected, inputs)  # torch.sum( outputs.to(torch.bool) == expected.to(torch.bool) ).cpu().numpy() / np.prod(outputs.shape)
            last_loss     = loss.item() / batch_size

            epoch_losses.append(last_loss)
            epoch_accuracies.append( last_accuracy )

            loop_loss   += last_loss
            loop_acc    += last_accuracy
            loop_count  += 1
            board_count += batch_size
            epoch_time   = time.perf_counter() - epoch_start

            # Print statistics after each epoch
            if board_count % 10_000 == 0:
                print(f'epoch: {epoch:4d} | board_count: {board_count:5d} | loss: {loop_loss/loop_count:.10f} | accuracy = {loop_acc/loop_count:.10f} | time: {1000*epoch_time/batch_size:.3f}ms/board')
                loop_loss  = 0
                loop_acc   = 0
                loop_count = 0

    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as exception:
        print(exception)
        raise exception
    finally:
        time_taken = time.perf_counter() - time_start
        print(f'Finished Training: {model.__class__.__name__} - {epoch} epochs in {time_taken:.1f}s')
        model.save()
        atexit.unregister(model.save)   # model now saved, so cancel atexit handler
        # model.eval()                  # disable dropout
