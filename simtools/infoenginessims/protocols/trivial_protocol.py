class TrivialProtocol:

    def __init__(self, param_vals, total_time=1., initial_time=0.):

        final_time = initial_time + total_time

        self.param_vals = param_vals
        self.total_time = total_time
        self.initial_time = initial_time
        self.final_time = final_time

    def get_param(self, param_index, time):

        param_val = self.param_vals[param_index]

        return param_val