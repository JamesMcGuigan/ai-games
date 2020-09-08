#!/usr/bin/env python3
import itertools
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from joblib import delayed
from joblib import Parallel

from core.ConnectXBBNN import *
from neural_networks.is_gameover.device import device
from neural_networks.is_gameover.IsGameoverCNN import isGameoverCNN

DatasetItem = namedtuple('DatasetItem', ['expected', 'bitboard'])
def generate_dataset(dataset_size: int, bitboard_fn, duplicates=2, verbose=False) -> Tuple[int, np.ndarray]:
    """ Creates a statistically balanced dataset of all the edgecases """
    time_start   = time.perf_counter()
    data = {
        "not_gameover":      [],
        "is_gameover":       [],
        "has_no_more_moves": []
    }
    dataset_size = int( dataset_size / duplicates / len(data) )  # 6 = (2 + 2 + 2)
    while min(map(len, data.values())) < dataset_size:
        def generate(games=100, agent=get_random_move):
            data      = defaultdict(list)
            bitboard  = empty_bitboard()
            player_id = current_player_id(bitboard)
            for n in range(games):
                while not is_gameover(bitboard):
                    action    = agent(bitboard)
                    bitboard  = result_action(bitboard, action, player_id)
                    mirror    = mirror_bitboard(bitboard)  # mirror here for random sampling or mirrors and bitboards
                    player_id = next_player_id(player_id)
                    if   has_no_more_moves(bitboard): [ data['has_no_more_moves'].append(x) for x in [ bitboard, mirror ] ]
                    elif is_gameover(bitboard):       [ data['is_gameover'      ].append(x) for x in [ bitboard, mirror ] ]
                    else:                             [ data['not_gameover'     ].append(x) for x in [ bitboard, mirror ] ]
            return data

        # batched_datas = [ generate() ]  # for debugging
        batched_datas = Parallel(n_jobs=os.cpu_count())(
            # [ delayed(generate)(100, get_random_move)      for round in range(100) ]  # 1 in 335 games are draws
            [ delayed(generate)(100, get_random_draw_move) for round in range(100) ]    # 1 in   7 games are draws
        )
        for (key, value), batched_data in itertools.product(data.items(), batched_datas):
            data[key] += batched_data[key]

    # has_no_more_moves are very rare, so duplicate datapoints, mirror and then resample from the rest
    bitboards = [
        *random.sample(data['not_gameover']      * duplicates,  dataset_size * duplicates),
        *random.sample(data['is_gameover']       * duplicates,  dataset_size * duplicates),
        *random.sample(data['has_no_more_moves'] * duplicates,  dataset_size * duplicates),
    ]
    np.random.shuffle(bitboards)
    output = [
        DatasetItem(bitboard_fn(bitboard), bitboard)
        for bitboard in bitboards
    ]
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
        # {'time_taken':  '77s', 'time_count': '391μs', 'count':  197574, 'not_gameover': 0.95293, 'is_gameover': 0.04694, 'has_no_more_moves': 0.00013}  # Kaggle Notebook with    @njit - 10% dataset
        # {'time_taken':  '67s', 'time_count': '473μs', 'count':  142134, 'not_gameover': 0.95286, 'is_gameover': 0.04696, 'has_no_more_moves': 0.00018}  # Kaggle Notebook without @njit - 10% dataset
    return output


def train(model, criterion, optimizer, bitboard_fn=is_gameover, dataset_size=1000, timeout=4*60*60):
    print(f'Training: {model.__class__.__name__}')
    time_start = time.perf_counter()
    epoch = 0
    try:
        model.load()
        hist_accuracy = [0]

        # dataset generation is expensive, so loop over each dataset until fully learnt
        while np.min(hist_accuracy[-10:]) < 1.0:  # need multiple epochs of 100% accuracy to pass
            if time.perf_counter() - time_start > timeout: break
            epoch         += 1
            epoch_start    = time.perf_counter()
            bitboard_count = 0

            dataset_epoch = 0
            dataset   = generate_dataset(dataset_size, bitboard_fn, verbose=True)
            expected  = np.array([ d.expected for d in dataset ])
            labels    = torch.tensor(expected, dtype=torch.float).to(device)  # nn.MSELoss() == float | nn.CrossEntropyLoss() == long
            bitboards = [ d.bitboard for d in dataset ]
            inputs    = torch.stack([ model.cast(bitboard) for bitboard in bitboards ]).to(device)

            while hist_accuracy[-1] < 1.0:      # loop until 100% accuracy on dataset
                if time.perf_counter() - time_start > timeout: break
                if dataset_epoch > 10: break  # if we plataeu after many iterations, then generate a new dataset

                dataset_epoch   += 1
                bitboard_count  += len(inputs)

                optimizer.zero_grad()
                outputs   = model(inputs)
                loss      = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                actual           = model.cast_from_outputs(outputs)  # convert (1,1) tensor back to bool
                running_accuracy = np.count_nonzero( actual == expected ) / len(actual)
                running_loss     = loss.item()
                running_count    = 1

                epoch_time       = time.perf_counter() - epoch_start
                last_loss        = running_loss     / running_count
                last_accuracy    = running_accuracy / running_count
                hist_accuracy.append(last_accuracy)

                # Print statistics after each epoch
                if dataset_epoch % 1 == 0:
                    print(f'epoch: {epoch:4d} | bitboards: {bitboard_count:5d} | loss: {last_loss:.5f} | accuracy: {last_accuracy:.5f} | time: {epoch_time :.0f}s')
            print(f'epoch: {epoch:4d} | bitboards: {bitboard_count:5d} | loss: {last_loss:.5f} | accuracy: {last_accuracy:.5f} | time: {epoch_time :.0f}s')
            model.save()


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

    # model     = isGameoverSquareNN
    # criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
    # optimizer = optim.Adadelta(model.parameters())
    # train(model, criterion, optimizer)

    model     = isGameoverCNN
    criterion = nn.MSELoss()  # NOTE: nn.CrossEntropyLoss() is for multi-output classification
    optimizer = optim.Adadelta(model.parameters())
    train(model, criterion, optimizer)

