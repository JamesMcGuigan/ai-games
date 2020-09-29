import time

import numpy as np

from constraint_satisfaction.fix_submission import is_valid_solution
from neural_networks.OuroborosLife import OuroborosLife
from utils.datasets import *
from utils.util import batch
from utils.util import csv_to_delta_list
from utils.util import csv_to_numpy_list
from utils.util import numpy_to_series


def ouroborors_dataframe(df: pd.DataFrame, filename=f'{output_directory}/submission_ouroborors.csv'):
    time_start    = time.perf_counter()
    model         = OuroborosLife()
    submission_df = sample_submission_df.copy()

    stats = {
        "boards":  { "solved": 0, "total": 0 },
        "delta":   { "solved": 0, "total": 0 },
        "dpixels": { "solved": 0, "total": 0 },
        "pixels":  { "solved": 0, "total": 0 },
    }
    for delta in range(1,5+1):
        df_delta = df[ df.delta == delta ]
        idxs   = csv_to_delta_list(df_delta)
        boards = csv_to_numpy_list(df_delta, key='stop')
        for idxs, inputs in zip(batch(idxs, 100), batch(boards, 100)):
            outputs = inputs
            for t in range(delta):
                outputs = model.predict(outputs)[:,0,:,:]
            for idx, output_board, input_board in zip(idxs, outputs, inputs):
                stats['boards']['total']   += 1
                stats['delta']['total']    += 1
                stats['pixels']['total']   += outputs.size
                stats['pixels']['solved']  += np.count_nonzero( outputs == inputs )
                stats['dpixels']['total']  += outputs.size
                stats['dpixels']['solved'] += np.count_nonzero( outputs == inputs )
                if is_valid_solution(output_board, input_board, delta):
                    stats['boards']['solved'] += 1
                    stats['delta']['solved']  += 1
                    submission_df.loc[idx]     = numpy_to_series(output_board, key='start')
        time_taken = time.perf_counter() - time_start
        print(f"delta = {delta} | solved {stats['delta']['solved']:4d}/{stats['delta']['total']} = {100*stats['delta']['solved']/stats['delta']['total']:4.1f}% | {100*stats['dpixels']['solved']/stats['dpixels']['total']:4.1f}% pixels | in {time_taken//60:.0f}:{time_taken%60:02.0f}")
        stats['delta']   = { "solved": 0, "total": 0 }
        stats['dpixels'] = { "solved": 0, "total": 0 }

    time_taken = time.perf_counter() - time_start
    print(f"ouroborors_dataframe() - solved {stats['boards']['solved']:4d}/{stats['boards']['total']} = {100*stats['boards']['solved']/stats['boards']['total']:4.1f}% | {100*stats['pixels']['solved']/stats['pixels']['total']:4.1f}% pixels | in {time_taken//60:.0f}:{time_taken%60:02.0f}")
    if filename: submission_df.to_csv(filename)
    return submission_df

if __name__ == '__main__':
    ouroborors_dataframe(test_df)
