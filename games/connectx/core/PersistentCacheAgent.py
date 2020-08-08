import atexit
import math

from util.base64_file import base64_file_load
from util.base64_file import base64_file_save


class PersistentCacheAgent:
    persist = False
    cache   = {}
    verbose = True

    def __new__(cls, *args, **kwargs):
        # noinspection PyUnresolvedReferences
        for parentclass in cls.__mro__:  # https://stackoverflow.com/questions/2611892/how-to-get-the-parents-of-a-python-class
            if cls is parentclass: continue
            if cls.cache is getattr(parentclass, 'cache', None):
                cls.cache = {}  # create a new cls.cache for each class
                break
        instance = object.__new__(cls)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__()
        if not self.persist: return  # disable persistent caching
        self.load()
        self.autosave()

    def autosave( self ):
        # Autosave on Ctrl-C
        atexit.unregister(self.__class__.save)
        atexit.register(self.__class__.save)

    # def __del__(self):
    #     self.save()

    @classmethod
    def filename( cls ):
        return f'./data/{cls.__name__}_base64'

    @classmethod
    def load( cls ):
        if not cls.persist: return  # disable persistent caching
        if cls.cache:       return  # skip loading if the file is already in class memory, empty dict is False
        filename = cls.filename()
        loaded   = base64_file_load(filename, vebose=cls.verbose)
        if loaded:  # cls.cache should not be set to None
            cls.cache = loaded

    @classmethod
    def save( cls ):
        if not cls.persist: return  # disable persistent caching
        # cls.load()                # update any new information from the file
        if cls.cache:
            filename = cls.filename()
            base64_file_save(cls.cache, filename, vebose=cls.verbose)


    @staticmethod
    def cache_size( data ):
        return sum([
            len(value) if isinstance(key, str) and isinstance(value, dict) else 1
            for key, value in data.items()
        ])

    @classmethod
    def reset( cls ):
        cls.cache = {}
        cls.save()


    ### Caching
    @classmethod
    def cache_function( cls, function, game, player_id, *args, **kwargs ):
        hash = (player_id, game)  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.cache:   cls.cache[function.__name__] = {}
        if hash in cls.cache[function.__name__]: return cls.cache[function.__name__][hash]

        score = function(game, *args, **kwargs)
        cls.cache[function.__name__][hash] = score
        return score

    @classmethod
    def cache_infinite( cls, function, game, player_id, *args, **kwargs ):
        # Don't cache heuristic values, only terminal states
        hash = (player_id, game)  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.cache:   cls.cache[function.__name__] = {}
        if hash in cls.cache[function.__name__]: return cls.cache[function.__name__][hash]

        score = function(game, player_id, *args, **kwargs)
        if abs(score) == math.inf: cls.cache[function.__name__][hash] = score
        return score
