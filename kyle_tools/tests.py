from .utilities import *
from .multisim import *
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
        assert type(d['nparray']) == list, 'failed to convert ndarray'
        assert type(d['nparray_list'][0]) == list, 'failed to convert ndarray in list'
        assert type(d['mix_list'][0]) == dict, 'failed to take dict -> dict in mixed list'
        assert type(d['mix_list'][0]['np_array']) == list, 'failed to convert ndarray in nested dict'

    if return_dicts:
        return dj, d

def test_numpify():
    dj, d = test_jsonify(return_dicts=True)
    assert numpify(dj) == d


class SimManagerTest(SimManager):
    def __init__(self):
        self.params = {k:v for k,v in zip(['position','sigma'], [[1,1],.01])}
        self.save_name = [None,'./test_output/']
        self.save_procs = [SaveSimOutput()]

    def initialize_sim(self):
        mu = self.params['position']
        sigma = self.params['sigma']
        self.sim = GaussianGenerator(mu, sigma)
        return 

    def analyze_output(self):
        pass

    def verify_param(self, key, val):
        if key == 'position':
            return True
        if key == 'sigma':
            return 0 < val


class SaveSimOutput():
    def run(self,SimManager):
        SimManager.save_dict.update({'output':SimManager.sim.output})

class GaussianGenerator():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma= sigma
    def run(self, **kwargs):
        out = np.random.normal(self.mu, self.sigma, np.shape(self.mu))
        try:
            return list(out)
        except:
            return out

space_filler = FillSpace(SimManagerTest(), param_keys=['position'])
random_filler = ParamGuider(SimManagerTest(), param_keys=['position'])

def trunc_func(val):
    return np.sum(np.square(val-1)) < 1

def val_func(simrun):
    return simrun.sim.output


setattr(space_filler, 'truncate_val', trunc_func)
setattr(space_filler, 'get_val', val_func)

setattr(random_filler, 'truncate_val', trunc_func)
setattr(random_filler, 'get_val', val_func)