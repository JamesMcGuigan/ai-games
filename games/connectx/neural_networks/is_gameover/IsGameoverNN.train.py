import time

import torch
import torch.nn as nn
import torch.optim as optim

from core.ConnectXBBNN import *
from neural_networks.is_gameover.IsGameoverSquareNN import isGameoverSquareNN

model     = isGameoverSquareNN
criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
optimizer = optim.Adadelta(model.parameters())

def train(model, criterion, optimizer, epochs=100):
    time_start = time.perf_counter()
    try:
        model.load()

        count_move       = 0
        last_accuracy    = 0
        running_accuracy = 0
        running_loss     = 0.0
        for count_game in range(epochs):
            if last_accuracy == 1.0: break

            bitboard    = empty_bitboard()
            player_id   = current_player_id(bitboard)
            while not is_gameover(bitboard):
                count_move += 1

                action    = get_random_move(bitboard)
                bitboard  = result_action(bitboard, action, player_id)
                player_id = next_player_id(player_id)
                expected  = is_gameover(bitboard)
                labels    = torch.tensor([[ expected ]], dtype=torch.float)  # nn.MSELoss() == float | nn.CrossEntropyLoss() == long

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(bitboard)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update running losses and accuracy
                actual            = bool( np.round( outputs.data.numpy().flatten()[0] ) )
                running_accuracy += int( actual == expected )
                running_loss     += loss.item()

                # Print statistics
                window = 100
                if count_move % window == 0:
                    last_accuracy = running_accuracy/window
                    print(f'game: {count_game+1:4d} | move: {count_move:5d}] | loss: {running_loss/window:.3f} | accuracy: {running_accuracy/window:.3f} ')
                    running_accuracy = 0
                    running_loss     = 0.0
                    break

    except: pass
    finally:
        time_taken = time.perf_counter() - time_start
        print(f'Finished Training: {epochs} epochs in {time_taken:.1f}s')
        model.save()


train(model, criterion, optimizer)