from numpy import empty


class Harmonic2D:

    def __init__(self, protocol,
                 spring_const0_scale, spring_const1_scale,
                 center0_scale, center1_scale, min_height_scale,
                 has_velocity=True):

        param_scales = [spring_const0_scale,
                        spring_const1_scale,
                        center0_scale,
                        center1_scale,
                        min_height_scale]

        self.protocol = protocol

        self.param_scales = param_scales

        self.has_velocity = has_velocity

    def _get_scaled_param_val(self, param_index, time):

        param_scale = self.param_scales[param_index]
        param_val = self.protocol.get_param_val(param_index, time)

        param_val_scaled = param_val * param_scale

        return param_val_scaled

    def get_potential(self, state, time, has_velocity=None):

        spring_const0 = self._get_scaled_param_val(0, time)
        spring_const1 = self._get_scaled_param_val(1, time)
        center0 = self._get_scaled_param_val(2, time)
        center1 = self._get_scaled_param_val(3, time)
        min_height = self._get_scaled_param_val(4, time)

        if has_velocity is None:
            has_velocity = self.has_velocity

        x0 = state[..., 0, 0] if has_velocity else state[..., 0]
        x1 = state[..., 1, 0] if has_velocity else state[..., 1]

        potential = min_height \
                    + 1/2 * (  spring_const0 * (x0 - center0) ** 2
                             + spring_const1 * (x1 - center1) ** 2)


        return potential

    def get_external_force(self, state, time, has_velocity=None):

        spring_const0 = self._get_scaled_param_val(0, time)
        spring_const1 = self._get_scaled_param_val(1, time)
        center0 = self._get_scaled_param_val(2, time)
        center1 = self._get_scaled_param_val(3, time)

        if has_velocity is None:
            has_velocity = self.has_velocity

        x0 = state[..., 0, 0] if has_velocity else state[..., 0]
        x1 = state[..., 1, 0] if has_velocity else state[..., 1]

        force0 = spring_const0 * (center0 - x0)
        force1 = spring_const1 * (center1 - x1)

        force_shape = state.shape[:-1] if has_velocity else state.shape
        force = empty(force_shape)
        force[..., 0] = force0
        force[..., 1] = force1

        return force