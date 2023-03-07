import datetime

import numpy as np


def group_apply(values, group_ids, func, multiarg=False, strout=False):
    if group_ids.ndim == 2:
        ix = np.lexsort([group_ids[:, i] for i in range(group_ids.shape[1])])
        sids = group_ids[ix]
        cuts = np.any(sids[1:] != sids[:-1], axis=1)
    else:
        ix = np.argsort(group_ids, kind='mergesort')
        sids = group_ids[ix]
        cuts = sids[1:] != sids[:-1]

    reverse = invert_argsort(ix)
    values = values[ix]

    if strout:
        nvalues = np.prod(values.shape)
        res = np.array([None]*nvalues).reshape(values.shape)
    elif multiarg:
        res = np.nan * np.zeros(len(values))
    else:
        res = np.nan * np.zeros(values.shape)

    prevcut = 0
    for cut in np.where(cuts)[0]+1:
        if multiarg:
            res[prevcut:cut] = func(*values[prevcut:cut].T)
        else:
            res[prevcut:cut] = func(values[prevcut:cut])
        prevcut = cut
    if multiarg:
        res[prevcut:] = func(*values[prevcut:].T)
    else:
        res[prevcut:] = func(values[prevcut:])
    revd = res[reverse]
    return revd


def invert_argsort(argsort_ix):
    reverse = np.repeat(0, len(argsort_ix))
    reverse[argsort_ix] = np.arange(len(argsort_ix))
    return reverse


def get_timestamp(time_format='%Y%m%d_%H%M%S'):
    """ Returns a timestamp by checking the date and time at the moment. """
    return str(datetime.datetime.utcnow().strftime(time_format))
