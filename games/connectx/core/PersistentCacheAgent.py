import atexit
import gzip
import math
import os
import pickle
import time
import zlib



class PersistentCacheAgent:
    debug   = not os.environ.get('KAGGLE_KERNEL_RUN_TYPE',None)
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
        if self.debug: return
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
        return './.cache/' + cls.__name__ + '.zip.pickle'

    @classmethod
    def load( cls ):
        if cls.debug: return  # remove caching on debug
        if cls.cache: return  # skip loading if the file is already in class memory
        try:
            # class cache may be more upto date than the pickle file, so avoid race conditions with multiple instances
            filename   = cls.filename()
            start_time = time.perf_counter()
            with gzip.GzipFile(filename, 'rb') as file:  # reduce filesystem cache_size
                # print("loading: "+cls.file )
                data = pickle.load(file)
                cls.cache.update({ **data, **cls.cache })
                if cls.verbose:
                    print("loaded: {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(
                        filename,
                        os.path.getsize(filename)/1024/1024,
                        time.perf_counter() - start_time,
                        cls.cache_size(cls.cache),
                    ))
        except (IOError, TypeError, EOFError, zlib.error) as exception:
            pass

    @classmethod
    def save( cls ):
        if cls.debug: return  # remove caching on debug
        # cls.load()          # update any new information from the file
        if cls.cache:
            filename = cls.filename()
            dirname  = os.path.dirname(filename)
            if not os.path.exists(dirname): os.mkdir(dirname)
            start_time = time.perf_counter()
            # print("saving: " + filename )
            with gzip.GzipFile(filename, 'wb') as file:  # reduce filesystem cache_size
                pickle.dump(cls.cache, file)
                if cls.verbose:
                    print("wrote:  {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(
                        filename,
                        os.path.getsize(filename)/1024/1024,
                        time.perf_counter() - start_time,
                        cls.cache_size(cls.cache),
                    ))

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
    def cache_function( cls, function, game, *args, **kwargs ):
        hash = game
        if function.__name__ not in cls.cache:   cls.cache[function.__name__] = {}
        if not cls.debug and hash in cls.cache[function.__name__]: return cls.cache[function.__name__][hash]

        score = function(game, *args, **kwargs)
        cls.cache[function.__name__][hash] = score
        return score

    @classmethod
    def cache_infinite( cls, function, game, *args, **kwargs ):
        # Don't cache heuristic values, only terminal states
        hash = game  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.cache:   cls.cache[function.__name__] = {}
        if not cls.debug and hash in cls.cache[function.__name__]: return cls.cache[function.__name__][hash]

        score = function(game, *args, **kwargs)
        if abs(score) == math.inf: cls.cache[function.__name__][hash] = score
        return score
