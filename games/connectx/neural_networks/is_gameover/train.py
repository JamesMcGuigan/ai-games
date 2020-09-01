#!/usr/bin/env python3
import itertools
import os
import random
import time
from collections import defaultdict

import torch.nn as nn
import torch.optim as optim
from joblib import delayed
from joblib import Parallel

from core.ConnectXBBNN import *
from neural_networks.is_gameover.IsGameoverCNN import isGameoverCNN
from neural_networks.is_gameover.IsGameoverSquareNN import isGameoverSquareNN


def generate_dataset(dataset_size: int, verbose=False) -> List[np.ndarray]:
    """ Creates a statistically balanced dataset of all the edgecases """
    time_start   = time.perf_counter()
    dataset_size = int(dataset_size // 4)
    data = {
        "not_gameover":      [],
        "is_gameover":       [],
        "has_no_more_moves": []
    }
    while min(map(len, data.values())) < dataset_size:
        def generate(games=100):
            data      = defaultdict(list)
            bitboard  = empty_bitboard()
            player_id = current_player_id(bitboard)
            for n in range(games):
                while not is_gameover(bitboard):
                    action    = get_random_move(bitboard)
                    bitboard  = result_action(bitboard, action, player_id)
                    player_id = next_player_id(player_id)
                    if   has_no_more_moves(bitboard): data['has_no_more_moves'].append(bitboard)
                    elif is_gameover(bitboard):       data['is_gameover'].append(bitboard)
                    else:                             data['not_gameover'].append(bitboard)
            return data
        # batched_datas = [ generate() ]  # for debugging
        batched_datas = Parallel(n_jobs=os.cpu_count())([ delayed(generate)(100) for _ in range(100) ])
        for (key, value), batched_data in itertools.product(data.items(), batched_datas):
            data[key] += batched_data[key]

    output = [
        *random.sample(data['not_gameover'], dataset_size),
        *random.sample(data['not_gameover'], dataset_size),
        *random.sample(data['is_gameover'],  dataset_size),
        *random.sample(data['has_no_more_moves'], dataset_size),
    ]
    np.random.shuffle(output)

    if verbose:
        time_taken = time.perf_counter() - time_start
        data_count = sum(map(len, data.values()))
        statistics = {
            "time_taken": f'{time_taken:.0f}s',
            "time_count": f'{time_taken/data_count*1000000:.0f}μs',
            "count": data_count,
            **{ key: round( len(value)/data_count, 5) for key, value in data.items() }
        }
        print('dataset statistics: ', statistics)
        # {'time_taken':  '84s', 'time_count':  '50μs', 'count': 1693800, 'not_gameover': 0.95312, 'is_gameover': 0.04673, 'has_no_more_moves': 0.00015}  # RazerBlade laptop
        # {'time_taken':  '77s', 'time_count': '391μs', 'count':  197574, 'not_gameover': 0.95293, 'is_gameover': 0.04694, 'has_no_more_moves': 0.00013}  # 7.8x slower - Kaggle with    @njit - 10% dataset
        # {'time_taken':  '67s', 'time_count': '473μs', 'count':  142134, 'not_gameover': 0.95286, 'is_gameover': 0.04696, 'has_no_more_moves': 0.00018}  # 9.5x slower - Kaggle without @njit - 10% dataset
    return output


def train(model, criterion, optimizer, dataset_size=1000, timeout=4*60*60):
    print(f'Training: {model.__class__.__name__}')
    time_start = time.perf_counter()
    epoch = 0
    try:
        model.load()

        hist_accuracy = [0]
        while np.min(hist_accuracy[-10:]) < 1.0:  # need multiple epochs of 100% accuracy to pass
            if time.perf_counter() - time_start > timeout: break

            epoch      += 1
            epoch_start = time.perf_counter()

            running_accuracy = 0
            running_loss     = 0.0
            running_count    = 0
            bitboard_count   = 0

            dataset = generate_dataset(dataset_size, verbose=(epoch<=2))
            for _ in range(32):  # dataset generation is expensive, so loop over each dataset multiple times
                random.shuffle(dataset)
                for bitboard in dataset:
                    bitboard_count += 1
                    expected = is_gameover(bitboard)
                    labels   = model.cast_to_labels(expected)

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


            # Print statistics after each epoch
            epoch_time       = time.perf_counter() - epoch_start
            last_loss        = running_loss     / running_count
            last_accuracy    = running_accuracy / running_count
            hist_accuracy.append(last_accuracy)
            print(f'epoch: {epoch:4d} | bitboards: {bitboard_count:5d} | loss: {last_loss:.5f} | accuracy: {last_accuracy:.5f} | time: {epoch_time :.0f}s')
            model.save()

    except KeyboardInterrupt:
        pass
    finally:
        time_taken = time.perf_counter() - time_start
        print(f'Finished Training: {model.__class__.__name__} - {epoch} epochs in {time_taken:.1f}s')
        model.save()



if __name__ == '__main__':

    model     = isGameoverSquareNN
    criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
    optimizer = optim.Adadelta(model.parameters())
    train(model, criterion, optimizer)

    model     = isGameoverCNN
    criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
    optimizer = optim.Adadelta(model.parameters())
    train(model, criterion, optimizer)

