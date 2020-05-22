from src_james.original.Solver import Solver



class DoNothingSolver(Solver):
    def action( self, grid, task=None, *args ):
        return grid
