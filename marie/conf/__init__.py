import os
import sys

import marie.conf.settings as conf

# create settings object corresponding to specified env
APP_ENV = os.environ.get("APP_ENV", "Dev")
print(f"APP_ENVX = {APP_ENV}")
_current = getattr(sys.modules["marie.conf.settings"], "{0}Config".format(APP_ENV))()

for atr in [f for f in dir(_current) if not "__" in f]:
    val = os.environ.get(atr, getattr(_current, atr))
    setattr(sys.modules[__name__], atr, val)


def as_dict():
    res = {}
    for atr in [f for f in dir(conf) if not "__" in f]:
        val = getattr(conf, atr)
        res[atr] = val
    return res
