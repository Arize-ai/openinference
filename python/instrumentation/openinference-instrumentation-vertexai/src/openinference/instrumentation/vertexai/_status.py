"""
We can't track all the functions that have been monkey-patched. Instead, all
monkey-patched functions are inactivated/activated when this global variable
is set to False/True.
"""

_IS_INSTRUMENTED = False
