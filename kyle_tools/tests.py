from .utilities import *
import numpy as np


def test_jsonify(return_dicts=False):
    d = {}
    t1 = 'test string'
    t2 = np.random.normal(0,1,3)
    t3 = 3
    t4 = [t2, t2, t2]
    t5 = {'np_array':t2, 'np list':t4, 'int':t3}
    t6 = [t5, t4, t2]
    keys = ['str', 'nparray', 'int', 'nparray_list', 'dict', 'mix_list']
    for k,v in zip(keys,[t1, t2, t3, t4, t5, t6]):
        d[k] = v
    dj = jsonify(d)
    
    if not return_dicts:
        assert type(d1['nparray']) == list, 'failed to convert ndarray'
        assert type(d1['nparray_list'][0]) == list, 'failed to convert ndarray in list'
        assert type(d1['mix_list'][0]) == dict, 'failed to take dict -> dict in mixed list'
        assert type(d1['mix_list'][0]['np_array']) == list, 'failed to convert ndarray in nested dict'

    if return_dicts:
        return dj, d

def test_numpify():
    dj, d = test_jsonify(return_dicts=True)
    assert numpify(dj) == d


    

