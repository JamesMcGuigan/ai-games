#!/usr/bin/env python3
import time

import torch.nn as nn
import torch.optim as optim

from core.ConnectXBBNN import *
from neural_networks.is_gameover.IsGameoverCNN import isGameoverCNN
from neural_networks.is_gameover.IsGameoverSquareNN import isGameoverSquareNN

# model     = isGameoverSquareNN
model     = isGameoverCNN
criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
optimizer = optim.Adadelta(model.parameters())


def train(model, criterion, optimizer, epochs=1000000):
    print(f'Training: {model.__class__.__name__}')
    time_start = time.perf_counter()
    count_game = 0
    try:
        model.load()

        last_accuracy    = 0
        running_accuracy = 0
        running_loss     = 0.0
        running_count    = 0
        count_move       = 0
        base_accuracy    = 0.953  # average random game lasts 22 moves
        for count_game in range(1, epochs+1):
            if last_accuracy == 1.0: break

            bitboard    = empty_bitboard()
            player_id   = current_player_id(bitboard)
            while not is_gameover(bitboard):
                count_move += 1

                action    = get_random_move(bitboard)
                bitboard  = result_action(bitboard, action, player_id)
                player_id = next_player_id(player_id)
                expected  = is_gameover(bitboard)
                labels    = model.cast_to_labels(expected)

                # normalize the distribution of training data = 1 gameover for every 1 non-gameover
                if not is_gameover(bitboard) and np.random.rand() < base_accuracy:
                    continue

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(bitboard)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update running losses and accuracy
                actual            = model.cast_from_outputs(outputs)  # convert (1,1) tensor back to bool
                running_accuracy += int( actual == expected )
                running_loss     += loss.item()
                running_count    += 1

            # Print statistics
            window = 10000
            if count_game % window == 0:
                last_loss     = running_loss     / running_count
                last_accuracy = running_accuracy / running_count
                base_accuracy = 1 - (count_game  / count_move )    # accuracy predicted by guessing no every time
                print(f'game: {count_game:4d} | move: {count_move:5d} | loss: {last_loss:.3f} | accuracy: {last_accuracy:.3f} / {base_accuracy:.3f}')
                running_accuracy = 0
                running_loss     = 0.0
                running_count    = 0
                if last_accuracy == 1.0: break

    except KeyboardInterrupt:
        pass
    finally:
        time_taken = time.perf_counter() - time_start
        print(f'Finished Training: {model.__class__.__name__} - {count_game} epochs in {time_taken:.1f}s')
        model.save()


if __name__ == '__main__':
    train(model, criterion, optimizer)