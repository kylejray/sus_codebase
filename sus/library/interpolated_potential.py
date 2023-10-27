import numpy as np
from ..protocol_designer import Potential
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

class InterpolatedPotential_1D(Potential):
    def __init__(
        self,
        key_points_func,
        N_params,
        default_params=None,
        relevant_domain=None,
        conservative=True,
        knots=True,
        k=1,
        s=0,
    ):
        """
        potential: func
            the potential energy function
        external_force: func
            the force function
        N_params: int
            the number of parameters that the force/potential energy need to give well defined answers
        N_dim: int
            number of dimensions the potential is over
        default_params = None or list
            if None, will set each default to 1
            if list (length N_params), list becomes the default values for each parameter
        relevant_domain: None or ndarray of dimension [2, N_dim]
            stores the relevant working domain of the potential, where we expect interesting dynamics to happen
            if None, uses -2,2 for all dimensions
            if ndarray, take the array to be [ [x1_min, x2_min,....], [x1_max, x2_max,...]]
        """
        self.key_points = key_points_func
        self.scale = 1
        self.conservative = conservative
        self.N_params = N_params
        self.N_dim = 1
        self.default_params = default_params
        self.k = k
        self.s = s
        self.knots = knots
        if relevant_domain is None:
            self.domain = np.asarray(
                (-2 * np.ones(self.N_dim), 2 * np.ones(self.N_dim))
            )
        else:
            self.domain = np.asarray(relevant_domain)
    
    def spline(self, params):
        if self.knots:
            x_keys, u_keys, knot_list = self.key_points(params)
            return LSQUnivariateSpline(x_keys, u_keys, knot_list, k=self.k)
        else:
            try:
                x_keys, u_keys, _ = self.key_points(params)
            except:
                x_keys, u_keys = self.key_points(params)
            return UnivariateSpline(x_keys, u_keys, k=self.k, s=self.s)

    def pot(self, x, params):
        U = self.spline(params)
        return U(x)
    
    def force(self, x, params):
        U = self.spline(params)
        return -U.derivative()(x)



def double_well_keys(params):
    k = 100
    dx=.05
    def x_points(params):
        x0, x1, d0, d1, w0, w1, _ = params
        left_well = [x0-w0/2-dx, x0-w0/2, x0+w0/2, (x0+w0/2)*.975]
        right_well = [(x1-w1/2)*.975, x1-w1/2, x1+w1/2, x1+w1/2+dx]
        return left_well + right_well

    def y_points(params):
        x0, x1, d0, d1, w0, w1, tilt = params
        slope=tilt/(x1-x0)
        if slope==0:
            left_well = [d0+k*dx, -d0, -d0, 0 ]
            right_well = [d1+k*dx, -d1, -d1, 0 ][::-1]
        else:
            left_well = [d0+k*dx, slope*w0/2, -slope*w0/2, -slope*(w0/2+dx) ]
            right_well = [-(x1-x0-w1/2-dx)*slope, -(x1-x0-w1/2-dx)*slope-d1, -(x1-x0-w1/2-dx)*slope-d1, d1+k*dx  ]
        return left_well + right_well

    return x_points(params), y_points(params)


pwl_default = [-1, 1, 1, 1, .5, .5, 0]
pwl_dom = [[-3.],[3.]]

DoubleWell = InterpolatedPotential_1D(double_well_keys, 7, default_params=pwl_default, relevant_domain=pwl_dom)

 
def triple_well_keys(params):
    '''
    params = xL,x0,xR, dL,d0,dR, wL,w0,wR
    '''
    k=100
    xL, x0, xR, dL, d0, dR, wL, w0, wR = params
    def well(x, d, w, bias=None):
        dx = w/10
        x_list = [x-w/2-dx, x-w/2, x, x+w/2, x+w/2+dx]
        knot_list = x_list[::2]
        u_list = [0 , -d, -d, -d, 0]
        d/dx
        if bias is 'left':
            x_list.insert(0, x-w/2-(1-dx))
            u_list.insert(0, k)
        if bias is 'right':
            x_list.append( x+w/2+(1-dx))
            u_list.append(k)
        return [x_list, u_list, knot_list]
    x_out, u_out, k_out = [], [], []
    for item in [well(xL, dL, wL, bias='left'), well(x0, d0, w0), well(xR, dR, wR, bias='right')]:
        x_out.extend(item[0])
        u_out.extend(item[1])
        k_out.extend(item[2])
    return [x_out, u_out, k_out]

twl_default = [-1, 0, 1, 2, 2, 2, .2, .2, .2]
twl_dom = [[-1.5],[1.5]]

TripleWell = InterpolatedPotential_1D(triple_well_keys, 9, default_params=twl_default, relevant_domain=twl_dom)



