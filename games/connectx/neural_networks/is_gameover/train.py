import time

import torch
import torch.nn as nn
import torch.optim as optim

from core.ConnectXBBNN import *
from neural_networks.is_gameover.IsGameoverSquareNN import isGameoverSquareNN

model     = isGameoverSquareNN
# model     = isGameoverCNN
criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
optimizer = optim.Adadelta(model.parameters())


def train(model, criterion, optimizer, epochs=1000):
    time_start = time.perf_counter()
    try:
        model.load()

        last_accuracy    = 0
        running_accuracy = 0
        running_loss     = 0.0
        running_count    = 0
        count_move       = 0
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
                running_count    += 1

            # Print statistics
            window = 10
            if count_game % window == 0 or count_game == epochs-1:
                last_loss     =     running_loss/running_count
                last_accuracy = running_accuracy/running_count
                print(f'game: {count_game:4d} | move: {count_move:5d} | loss: {last_loss:.3f} | accuracy: {last_accuracy:.3f} ')
                running_accuracy = 0
                running_loss     = 0.0
                running_count    = 0

    except KeyboardInterrupt:
        pass
    finally:
        time_taken = time.perf_counter() - time_start
        print(f'Finished Training: {epochs} epochs in {time_taken:.1f}s')
        model.save()


if __name__ == '__main__':
    train(model, criterion, optimizer)