from src.core.DataModel import Competition
from src.core.DataModel import Dataset
from src.settings import settings

if __name__ == '__main__':

    print('\n','-'*20,'\n')
    print('Abstraction and Reasoning Challenge')
    print('Team: Mathematicians + Experts')
    print('https://www.kaggle.com/c/abstraction-and-reasoning-challenge')
    print('\n','-'*20,'\n')

    # This is the actual competition submission entry
    # Do this first incase we have notebook timeout issues (>9h runtime)
    test_dir = f"{settings['dir']['data']}/test"
    dataset  = Dataset(test_dir, 'test')
    dataset.solve()
    for key, value in dataset.score().items():
        print( f"{key:11s}: {value}" )
    dataset.write_submission()

    print('\n','-'*20,'\n')

    # Then run the script against public competition data
    # This is mostly just to make it easier to see the stats in the published notebook logs
    competition = Competition()
    competition.solve()
    for key, value in competition.score().items():
        print( f"{key:11s}: {value}" )
