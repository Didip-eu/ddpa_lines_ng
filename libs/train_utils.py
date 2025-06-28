import time
import random

def split_set( *arrays, limit=0, reserved_ratio=.2, random_state=46):
    random.seed( random_state)
    seq = range(len(arrays[0]))
    train_set = set(random.sample( seq, int(len(arrays[0])*(1-reserved_ratio))))
    reserved_set = set(seq) - train_set
    if limit:
        assert limit <= len(arrays[0])
        train_set=set(random.sample( seq, limit))
    sets = []
    for a in arrays:
        sets.extend( [[ a[i] for i in train_set ], [ a[j] for j in reserved_set ]] )
    return sets

def duration_estimate( iterations_past, iterations_total, current_duration ):
    time_left = time.gmtime((iterations_total - iterations_past) * current_duration)
    return ''.join([
     '{} d '.format( time_left.tm_mday-1 ) if time_left.tm_mday > 0 else '',
     '{} h '.format( time_left.tm_hour ) if time_left.tm_hour > 0 else '',
     '{} mn'.format( time_left.tm_min ) ])


