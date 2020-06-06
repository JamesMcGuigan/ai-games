import atexit
import gzip
import math
import os
import pickle
import time

from sample_players import BasePlayer



class DataSavePlayer(BasePlayer):
    data = {}

    def __new__(cls, *args, **kwargs):
        for parentclass in cls.__mro__:  # https://stackoverflow.com/questions/2611892/how-to-get-the-parents-of-a-python-class
            if cls is parentclass: continue
            if cls.data is getattr(parentclass,'data',None):
                cls.data = {}  # create a new cls.data for each class
                break
        instance = object.__new__(cls)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        return './data/' + cls.__name__ + '.zip.pickle'

    @classmethod
    def load( cls ):
        if cls.data: return  # skip loading if the file is already in class memory
        try:
            # class data may be more upto date than the pickle file, so avoid race conditions with multiple instances
            filename   = cls.filename()
            start_time = time.perf_counter()
            with gzip.GzipFile(filename, 'rb') as file:  # reduce filesystem size
                # print("loading: "+cls.file )
                data = pickle.load(file)
                cls.data.update({ **data, **cls.data })
                print("loaded: {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(
                    filename,
                    os.path.getsize(filename)/1024/1024,
                    time.perf_counter() - start_time,
                    cls.size(cls.data),
                ))
        except (IOError, TypeError, EOFError) as exception:
            pass

    @classmethod
    def save( cls ):
        # cls.load()  # update any new information from the file
        if cls.data:
            filename   = cls.filename()
            start_time = time.perf_counter()
            print("saving: " + filename )
            with gzip.GzipFile(filename, 'wb') as file:  # reduce filesystem size
                pickle.dump(cls.data, file)
                print("wrote:  {:40s} | {:4.1f}MB in {:4.1f}s | entries: {}".format(
                    filename,
                    os.path.getsize(filename)/1024/1024,
                    time.perf_counter() - start_time,
                    cls.size(cls.data),
                ))

    @staticmethod
    def size( data ):
        return sum([
            len(value) if isinstance(key, str) and isinstance(value, dict) else 1
            for key, value in data.items()
        ])

    @classmethod
    def reset( cls ):
        cls.data = {}
        cls.save()


    ### Caching
    @classmethod
    def cache(cls, function, state, player_id, *args, **kwargs):
        hash = (player_id, state)  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.data:   cls.data[function.__name__] = {}
        if hash in cls.data[function.__name__]: return cls.data[function.__name__][hash]

        score = function(state, player_id, *args, **kwargs)
        if abs(score) == math.inf: cls.data[function.__name__][hash] = score
        return score

    @classmethod
    def cache_infinite(cls, function, state, player_id, *args, **kwargs):
        # Don't cache heuristic values, only terminal states
        hash = (player_id, state)  # QUESTION: is player_id required for correct caching between games?
        if function.__name__ not in cls.data:   cls.data[function.__name__] = {}
        if hash in cls.data[function.__name__]: return cls.data[function.__name__][hash]

        score = function(state, player_id, *args, **kwargs)
        if abs(score) == math.inf:
            cls.data[function.__name__][hash] = score
        return score
