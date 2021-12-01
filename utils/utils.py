import os
import time


def ensure_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)   
    return dir   

def current_milli_time():
    return round(time.time() * 1000)   