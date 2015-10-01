import os
import datetime

def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def file_modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)
