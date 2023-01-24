from .utilities import save_as_json
import datetime
import random
import numpy as np

class SimManager:
    '''
    Mostly abtract container to hold methods of  initializing, running, and analyzing a simulation and its output, and saving the results.

    Attributes
    ----------
    params : dict
        for the value of all the parameters needed.
    save_procs : list
        procedures to do during the saving process; each should have a run method that writes stuff to a save_dict dictionary
    save_name : list
        len=2; first element is a file path to save and the second is the actual name

    '''

    def initialize_sim(self):
        '''
        This function should be defined to use self.params set an self.attribute sim with a run method that returns a simulation output
        '''
        pass
    
    def analyze_output(self):
        '''
        This function should be defined to work with a self.sim.output method to analyze the results. Meant to work with the save_procs attribute
        '''
        pass

    def run_sim(self, verbose=True, **sim_kwargs):
        '''
        Sets up a dictionary to save data in, and then initializes a simulation, runs it, and analyzes the output.

        '''
        self.save_dict={}
        self.save_dict['start_date'] = datetime.datetime.now()
        if verbose:
            print('\n initializing...')
        self.initialize_sim()
        if verbose:
            print('\n running sim...')
        self.sim.output = self.sim.run(**sim_kwargs)
        if verbose:
            print('\n analyzing output...')
        self.analyze_output()

    def verify_param(self, key, value):
        '''
        Should be defined to return False for any key,value pair that is not a valid parameter for the key; True otherwise
        '''
        return True

    def change_params(self, param_dict):
        '''
        Updates the self.params dictionary with the input parameter "param_dict"
        '''
        self.params.update(param_dict)
    
    
    def perturb_params(self, std=.1, n=1, which_params=None, verbose=False):
        '''
        A method to randomly change some of the values in the self.params dictionary, using a gaussian centered at the current value. Check the validity of each parameter through self.verify_param before accepting the jump.

        Parameters
        ----------
        std: float
            the scale of the standard deviation of the gaussian, .1 means 10% of the current value, for example
        n: int
            how many parameters to change
        which_params: list
            the keys of the parameters from which n will be selected, 'None' actually defaults to all
        varbose: bool
            wether to give feedback during the process
        '''
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
                new_val = np.random.normal(current_val, np.abs(std*current_val))
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
        ''''
        Sets up a save_dict if not already existing; then runs each save procedure. Save procedures should each have a run method that writes something to self.save_dict.
        '''
        if not hasattr(self, 'save_dict'):
            self.save_dict={}
        for item in self.save_procs:
            item.run(self,)
    
    def save_sim(self):
        '''
        Runs the save procedures, and saves the result. Name will be generated automatically if no self.save_name is found
        '''
        self.run_save_procs()
        try: save_name = self.save_name(self)
        except: save_name = self.save_name
        save_as_json(self.save_dict, *save_name)


class SaveProc():

    '''abstract class showing how to make a SaveProc'''

    def run(self, SimManager):
        SimManager.save_dict.update({'foo':'bar'})



class ParamGuider():
    '''
    A higher level container to guide a sweep or optimization on parameter space using instances of the SimManager class

    Attributes
    ----------
    SimManager: Class
        an instance of SimManager
    param_keys: list
        list of str, one for each parameter that will be involved in the optimization
    current_params: dict
        the current parameter values, initialized from SimManager.params but can depart from it
    val_list: list
        all the previously accepted values, not parameters but figure of merit (FOM) values extracted from sims
    verbose: bool
        wether certain things print or not
    '''
    def __init__(self, SimManager, param_keys=None):
        '''
        Parameters
        ----------
        SimManager: SimManager instace
        param_keys: list
        '''
        self.SimManager = SimManager
        if param_keys != None:
            self.param_keys = param_keys
        else:
            self.params_keys = list(SimManager.params.keys())
        self.current_params = {k:self.SimManager.params[k] for k in self.param_keys }
        self.verbose = False
        self.val_list = []

    def get_current_val(self):
        '''
        A function that needs to be defined on a case by case basis. Will be used to extract a 'figure of merit'(FOM) after a sim is run to evaluate the parameter jump
        '''
        return self.get_val(self.SimManager)

    def get_prob(self, new_val, curr_val):
        '''
        This is the probability that a particular jump is accepted based on the current and the new FOM. Here the default is all jumps accepted
        '''
        return 1

    def truncate_val(self, new_val):
        '''
        This is to handle outcome values for the FOM that should be outright rejected. Default is to accept everything
        '''
        return True

    def iterate(self, curr_val, save=True, **kwargs):
        '''
        One iteration of the ParamGuiders main function. Perturbs the parameters, runs a sim, and then decides wether to accept the new peeturned parameters or not.

        Parameters
        ----------
        curr_val: float
            whatever the FOM was for the current parameter values
        save: bool
            wether to save each iteration as its own dictionary
        **kwargs: dict
            keyword arguments to be passed to perturn params, deciding how many parameters to perturb etc...
        
        Returns
        -------
        val: float
            either the new accepted value or the current value of the FOM
        True/False: bool
            returns True if the jump was accepted and False otherwise
        '''
        sm = self.SimManager
        sm.change_params(self.current_params)
        sm.perturb_params(which_params = self.param_keys, **kwargs)

        sm.run_sim(verbose=self.verbose)
        if save:
            sm.save_sim()
        new_val = self.get_current_val()
        accept_prob = self.get_prob(new_val, curr_val)

        if self.truncate_val(new_val) and np.random.uniform() < accept_prob :
            self.current_params =  sm.params.copy()
            self.val_list.append(new_val)
            if self.verbose:
                print(f'accepted new vals:{new_val} with prob{accept_prob:.2}')
            return new_val, True
        else:
            if self.verbose:
                print(f'rejected jump:{new_val} with prob{accept_prob:.2}')
            return curr_val, False
   
    

    def run(self, max_jumps=10, max_tries=100, **kwargs):
        '''
        Runs a maximum of max_tries iterations or until max_jumps parameter changes are accepted

        Parameters
        ----------
        max_jumps: int
            maximum number of times new parameters will be accepted before termination
        max_tries: int
            the maximum number of new parameter sets are tried before termination
        **kwargs: dict
            these are the iterate keywords, ultimately these are passed to the perturb_params method of SimManager
        '''
        sm = self.SimManager

        i = 0
        j = 0
        curr_i =0

        while j <= max_jumps and i <= max_tries:
            if i ==0:
                sm.change_params(self.current_params)
                sm.run_sim(verbose=self.verbose)
                sm.save_sim()
                curr_val = self.get_current_val()
                self.val_list.append(curr_val)
                if self.verbose:
                    print(f'initial vals:{curr_val}')
                
            if i > 0:
                curr_val, jump = self.iterate(curr_val, **kwargs)
                if jump:
                    if self.verbose:
                        print(f'after {i-curr_i} tries')
                    curr_i = i
                    j += 1 
            i += 1

                
class FillSpace(ParamGuider):
    '''
    Attempts to guide the parameters to spread out in "FOM" space by defining a get_prob function that acts like a repulsive 1/r potetial between all previously accepted values. Work still in progress, underperforming currently...
    '''
    def get_prob(self, new_val, old_val):
        ener = 0
        ener_old = 0
        try:
            past_vals = self.val_list
        except:
            self.val_list = [old_val]
            past_vals = self.val_list
        for val in past_vals :
            ener += np.sum(np.subtract(new_val, val)**2)
            ener_old += np.sum(np.subtract(old_val, val)**2)
        beta = 1
        if hasattr(self,'beta'):
            beta = self.beta

        return np.exp(-beta*ener_old/ener)
