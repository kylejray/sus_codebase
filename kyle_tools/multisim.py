from .utilities import save_as_json
import datetime
import random
import numpy as np

class SimManager:

    def initialize_sim(self):
        pass
    
    def analyze_output(self):
        pass

    def run_sim(self, **kwargs):
        self.save_dict={}
        self.save_dict['start_date'] = datetime.datetime.now()
        print('\n initializing...')
        self.initialize_sim()
        print('\n running sim...')
        self.sim.output = self.sim.run(**kwargs)
        print('\n analyzing output...')
        self.analyze_output()

    def change_params(self, param_dict):
        self.params.update(param_dict)
    
    
    def perturb_params(self, std=.1, n=1, which_params=None):
        if which_params is None:
            which_params = list(self.params)
        keys = random.choices(which_params, k=n)
        for key in keys:
            bool = True
            while bool:
                current_val = self.params[key]
                new_val = np.random.normal(current_val, std*current_val)
                if self.verify_param(key, new_val):
                    self.change_params({key:new_val})
                    bool = False


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