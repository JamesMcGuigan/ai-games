import cv2
import skimage.measure

from src_james.solver_multimodel.queries.ratio import task_shape_ratios
from src_james.solver_multimodel.Solver import Solver


class ZoomSolver(Solver):
    verbose = False

    def detect(self, task):
        ratios = task_shape_ratios(task)
        ratio  = list(ratios)[0]
        detect = (
                ratios != { (1,1) }   # not no scaling
                and len(ratios) == 1      # not multiple scalings
                and ratio[0] == ratio[1]  # single consistant scaling
        )
        return detect

    def get_scale(self, task):
        return task_shape_ratios(task)[0][0]

    def solve_grid(self, grid, task=None, *args):
        scale = self.get_scale(task)
        if scale > 1:
            resize = tuple( int(d*scale) for d in grid.shape )
            output = cv2.resize(grid, resize, interpolation=cv2.INTER_NEAREST)
        else:
            resize = tuple( int(1/scale) for d in grid.shape )
            output = skimage.measure.block_reduce(grid, resize)
        if self.verbose:
            print('scale', scale, 'grid.shape', grid.shape, 'output.shape', output.shape)
            print('grid', grid)
            print('output', output)
        return output
