from xgboost import XGBClassifier

from src.solver_multimodel.core.ProblemSetEncoder import ProblemSetEncoder


class XGBEncoder(ProblemSetEncoder):
    # DOCS: https://xgboost.readthedocs.io/en/latest/parameter.html
    # See:  src/solver_multimodel/XGBGridSolver.hyperopt.py
    # Be very conservative here as this is an inheritable class
    encoder_defaults = {
        **ProblemSetEncoder.encoder_defaults,
        'eval_metric': 'error',
        'n_jobs':      -1,
        # 'objective':       'reg:squarederror',
        # 'max_delta_step':   1,
        # 'max_depth':        1,
        # 'min_child_weight': 0,
        # 'num_classes':     11,
        # 'n_estimators':    1,
        # 'max_depth':       1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_encoder(self):
        encoder = XGBClassifier(**self.encoder_args)
        return encoder
