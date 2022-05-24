from .utilities import save_as_json

class SimManager:

    def initialize_sim(self):
        pass
    
    def analyze_output(self):
        pass

    def run_sim(self, **kwargs):
        print('\n initializing...')
        self.initialize_sim()
        print('\n running sim...')
        self.sim.output = self.sim.run(**kwargs)
        print('\n analyzing output...')
        self.analyze_output()

    def change_params(self, param_dict):
        self.params.update(param_dict)

    def run_save_procs(self):
        self.save_dict={}
        for item in self.save_procs:
            item.run(self,)
    
    def save_sim(self):
        self.run_save_procs()
        try: save_name = self.save_name(self)
        except: save_name = self.save_name
        save_as_json(self.save_dict, *save_name)