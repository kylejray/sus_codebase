class SimpleLinearProtocol:

    def __init__(self, param_initial_vals, param_final_vals,
                 total_time=1., initial_time=0.):

        final_time = initial_time + total_time

        self.param_initial_vals = param_initial_vals
        self.param_final_vals = param_final_vals
        self.total_time = total_time
        self.initial_time = initial_time
        self.final_time = final_time

    def get_param_val(self, param_index, time):

        param_initial_val = self.param_initial_vals[param_index]
        param_final_val = self.param_final_vals[param_index]
        total_time = self.total_time

        progress = time / total_time

        param_val = param_initial_val * (1 - progress) \
                + param_final_val * progress

        return param_val