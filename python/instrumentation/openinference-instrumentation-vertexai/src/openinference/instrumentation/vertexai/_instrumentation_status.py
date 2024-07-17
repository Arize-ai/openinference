"""
We can't track all the functions that have been monkey-patched. Instead, all
monkey-patched functions are inactivated or activated by setting the global
variable `_IS_INSTRUMENTED` to False or True, respectively.
"""

_IS_INSTRUMENTED = False
