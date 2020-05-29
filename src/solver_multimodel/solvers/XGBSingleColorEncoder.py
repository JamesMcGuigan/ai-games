from src.datamodel.Problem import Problem
from src.datamodel.ProblemSet import ProblemSet
from src.datamodel.Task import Task
from src.functions.queries.grid import *
from src.functions.transforms.singlecolor import identity
from src.functions.transforms.singlecolor import np_bincount
from src.functions.transforms.singlecolor import np_hash
from src.functions.transforms.singlecolor import np_shape
from src.functions.transforms.singlecolor import unique_colors_sorted
from src.solver_multimodel.core.XGBEncoder import XGBEncoder


class SingleColorXGBEncoder(XGBEncoder):
    dtype    = np.int8,
    encoders = {
        np.array: [ identity ],
    }
    features = {
        np.array:   [
            unique_colors_sorted,
            max_color,
            max_color_1d,
            min_color,
            min_color_1d,
            np_bincount,
            np_hash,
            np_shape,
            count_colors,
            count_squares,
        ],
        ProblemSet: [],
        Problem:    [],
        Task:       [
            # task_output_unique_sorted_colors
        ]
    }
    encoder_defaults = {
        **XGBEncoder.encoder_defaults,
        'max_delta_step':   np.inf,  # unsure if this has an effect
        'max_depth':        1,       # possibly required for this problem
        'n_estimators':     1,       # possibly required for this problem
        'min_child_weight': 0,       # possibly required for this problem
        # 'max_delta_step':   1,
        # 'objective':       'rank:map',
        # 'objective':       'reg:squarederror',
        # 'max_delta_step':   1,
        # 'n_jobs':          -1,
    }
    def __init__(self, encoders=None, features=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = encoders if encoders is not None else self.__class__.encoders
        self.features = features if features is not None else self.__class__.features


    ### DEBUG
    # def predict(self,
    #             problemset: Union[ProblemSet, Task],
    #             xgb:   XGBClassifier = None,
    #             task:  Task = None,
    #             *args, **kwargs
    # ) -> Any:
    #     task       = task or (problemset if isinstance(problemset, Task) else problemset.task)
    #     problemset = (problemset['test'] if isinstance(problemset, Task) else problemset )
    #     if task.filename not in self.cache:   self.fit(task)
    #     # if self.cache[task.filename] is None: return None  # Unsolvable mapping
    #
    #     output = [ 8 ] * len(task['test'])
    #     return output
