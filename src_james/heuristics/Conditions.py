import numpy as np

class GridConditions:

    @staticmethod
    def same_shape(input: np.ndarray, output: np.ndarray) -> bool:
        return input.shape == output.shape

    @staticmethod
    def rotated_shape(input: np.ndarray, output: np.ndarray) -> bool:
        """shape of axes have been rotated: (2,1) -> (1,2)"""
        return input.shape[0] == output.shape[1] and input.shape[1] == output.shape[0]

    @staticmethod
    def same_number_of_colors(input: np.ndarray, output: np.ndarray) -> bool:
        """both grids share the same colormap"""
        return np.unique(input) == np.unique(output)

    @staticmethod
    def same_pixels(input: np.ndarray, output: np.ndarray) -> bool:
        """pixel counts for both grids are the same"""
        return all(np.bincount(input) == np.bincount(output))



# class GridSizeConditions:
#     @classmethod
#     def task_shape_ratios(cls, task):
#         ratios = set([
#             cls.grid_shape_ratio(spec.get('input',[]), spec.get('output',[]))
#             for spec in cls.loop_specs(task, 'train')
#         ])
#         # ratios = set([ int(ratio) if ratio.is_integer() else ratio for ratio in chain(*ratios) ])
#         return ratios
#
#     @classmethod
#     def is_task_shape_ratio_unchanged(cls, task):
#         return cls.task_shape_ratios(task) == { (1,1) }
#
#     @classmethod
#     def is_task_shape_ratio_consistent(cls, task):
#         return len(cls.task_shape_ratios(task)) == 1
#
#     @classmethod
#     def is_task_shape_ratio_integer_multiple(cls, task):
#         ratios = cls.task_shape_ratios(task)
#         return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])
