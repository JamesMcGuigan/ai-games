import signal
from collections import Callable



def call_with_timeout( time_limit: float, function: Callable, *args, **kwargs ):
    # time.sleep(0)   # If any other process wants to interrupt us, do it now
    if time_limit:
        def raise_timeout(signum, frame): raise TimeoutError    # DOC: https://docs.python.org/3.6/library/signal.html
        signal.signal(signal.SIGPROF, raise_timeout)            # Register function to raise a TimeoutError on signal
        signal.setitimer(signal.ITIMER_PROF, time_limit)        # Schedule the signal to be sent after time_limit in milliseconds

    try:
        output = function(*args, **kwargs)
        signal.setitimer(signal.ITIMER_PROF, 0)                 # Unregister signal
        return output
    except TimeoutError as err:
        return TimeoutError


def call_with_timeout_ms( time_limit, function, *args, **kwargs ):
    return call_with_timeout(time_limit*1000, function, *args, **kwargs)
