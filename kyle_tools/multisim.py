from .utilities import save_as_json
import datetime
import random
import numpy as np

class SimManager:

    def initialize_sim(self):
        pass
    
    def analyze_output(self):
        pass

    def run_sim(self, verbose=True, **kwargs):
        self.save_dict={}
        self.save_dict['start_date'] = datetime.datetime.now()
        if verbose:
            print('\n initializing...')
        self.initialize_sim()
        if verbose:
            print('\n running sim...')
        self.sim.output = self.sim.run(**kwargs)
        if verbose:
            print('\n analyzing output...')
        self.analyze_output()

    def change_params(self, param_dict):
        self.params.update(param_dict)
    
    
    def perturb_params(self, std=.1, n=1, which_params=None, verbose=False):
        if which_params is None:
            which_params = list(self.params)
        keys = random.choices(which_params, k=n)
        for key in keys:
            i=0
            if verbose:
                print(f'changing param {key}')
            bool = True
            while bool and i < 1_000:
                i += 1
                current_val = self.params[key]
                new_val = np.random.normal(current_val, std*current_val)
                if verbose:
                    print(f'trial_value: {new_val}')
                if self.verify_param(key, new_val):
                    self.change_params({key:new_val})
                    bool = False
                    if verbose:
                        print('sucess')
                else:
                    if verbose:
                        print('failure')
            if i >= 1_000:
                print(f'gave up on param {key} after {i} tries')



    def run_save_procs(self):
        if not hasattr(self, 'save_dict'):
            self.save_dict={}
        for item in self.save_procs:
            item.run(self,)
    
    def save_sim(self):
        self.run_save_procs()
        try: save_name = self.save_name(self)
        except: save_name = self.save_name
        save_as_json(self.save_dict, *save_name)

class ParamGuider():
    def __init__(self, SimManager, param_keys=None):
        self.SimManager = SimManager
        if param_keys != None:
            self.param_keys = param_keys
        else:
            self.params_keys = list(SimManager.params.keys())
        self.current_params = {k:self.SimManager.params[k] for k in self.param_keys }
        self.verbose = False

    def get_current_val(self):
        return self.get_val(self.SimManager)

    def get_prob(self, new_val):
        return 1

    def truncate_val(self, new_val):
        return True

    def iterate(self, curr_val, save=True, **kwargs):
        sm = self.SimManager
        sm.change_params(self.current_params)
        sm.perturb_params(which_params = self.param_keys, **kwargs)

        sm.run_sim(verbose=self.verbose)
        if save:
            sm.save_sim()
        new_val = self.get_current_val()

        if self.truncate_val(new_val) and np.random.uniform() < self.get_prob(new_val, curr_val):
            self.current_params =  sm.params.copy()
            return new_val, True
        else:
            return curr_val, False
   
    

    def run(self, max_jumps=10, max_tries=100, **kwargs):
        sm = self.SimManager
        val_list = []
        i = 0
        curr_i =0

        while len(val_list) <= max_jumps and i <= max_tries:
            if i ==0:
                sm.run_sim()
                sm.save_sim()
                curr_val = self.get_current_val()
                val_list.appen(cur_val)
                if self.verbose:
                    print(f'initial vals:{curr_val}')
                
            if i > 0:
                curr_val, jump = self.iterate(curr_val, **kwargs)
                if jump:
                    val_list.append(curr_val)
                    if self.verbose:
                        print(f'accepted new vals:{curr_val} after {i-curr_i} tries')
                    curr_i = i
                else:
                    print(f'rejected jump:{curr_val}') 
            i += 1

                
class FillSpace(ParamGuider):

    def get_prob(self, new_val, old_val):
        ener = 0
        ener_old = 0
        try:
            past_vals = self.past_vals
        except:
            self.past_vals=[old_val]
            past_vals = self.past_vals
        for val in past_vals :
            ener += np.sum(np.subtract(new_val, val)**2)
            ener_old += np.sum(np.subtract(old_val, val)**2)
        
        return np.exp(-ener_old/ener)
