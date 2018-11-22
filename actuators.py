import numpy as np
from math_utils import cross
from scipy.stats import norm, uniform


class Actuators(object):
    def __init__(self,
                 rxwl_mass,
                 rxwl_radius,
                 w_rxwls=np.array([0, 0, 0]),
                 rxwl_max_torque=np.inf,
                 noise_factor=0.01):
        """Constructs an object to store reaction wheel state and methods
        
        Args:
            rxwl_mass (float): the mass of the reaction wheel (kg)
            rxwl_radius (float): the radius of the reaction wheel (m)
            w_rxwls (numpy ndarray, optional): Defaults to np.array([0, 0, 0]).
                The starting angular velocity of the x, y, and z reaction
                wheels.
            rxwl_max_torque (float, optional): Defaults to np.inf. The maximum
                torque (N * m) that a given reaction wheel can apply. If
                infinity, there is no limit.
            noise_factor (float, optional): Defaults to 0.01. The standard
                deviation of the Gaussian noise distribution centered at 0.
                Used to apply noise to the actuation of control torques.
        """
        self.C_w = 0.5 * rxwl_mass * rxwl_radius**2
        self.w_rxwls = w_rxwls
        self.rxwl_max_torque = rxwl_max_torque
        noise_func = norm(loc=0.0, scale=noise_factor)
        self.noise_vals = noise_func.rvs(size=100000, random_state=123456)

    def apply_control_torques(self, M_ctrl, w_sc, t, delta_t):
        """Applies the control torques to the modeled reaction wheels
        
        Args:
            M_ctrl (numpy ndarray): the control torque (3x1) produced by the
                PD controller (N * m)
            w_sc (numpy ndarray): the angular velocity (rad/s) (3x1) in body
                coordinates of the spacecraft (at a given time)
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            numpy ndarray: the control moment (3x1) as actually applied on
                the reaction wheels (the input control torque with some
                Gaussian noise applied) (N * m)
            numpy ndarray: the angular acceleration of the 3 reaction wheels
                applied to achieve the applied torque (rad/s^2)
        """
        # take into account the fact that reaction wheels can only apply a certain max torque
        M_ctrl_fixed = np.empty((3, ))
        M_ctrl_fixed[0] = np.sign(M_ctrl[0]) * min(
            abs(M_ctrl[0]), self.rxwl_max_torque)
        M_ctrl_fixed[1] = np.sign(M_ctrl[1]) * min(
            abs(M_ctrl[1]), self.rxwl_max_torque)
        M_ctrl_fixed[2] = np.sign(M_ctrl[2]) * min(
            abs(M_ctrl[2]), self.rxwl_max_torque)

        w_dot_rxwls = -cross(w_sc, self.w_rxwls) - 1 / self.C_w * M_ctrl_fixed
        w_dot_rxwls[0] = self.add_noise(w_dot_rxwls[0], t, delta_t)
        w_dot_rxwls[1] = self.add_noise(w_dot_rxwls[1], t, delta_t)
        w_dot_rxwls[2] = self.add_noise(w_dot_rxwls[2], t, delta_t)
        M_applied = -self.C_w * w_dot_rxwls - cross(w_sc,
                                                    self.C_w * self.w_rxwls)
        return M_applied, w_dot_rxwls

    def add_noise(self, value, t, delta_t):
        """Adds Gaussian noise to a given value

        NOTE: This method uses a cache of random values generated in this
              class's constructor. This is done to (1) reduce the overhead
              of many individual `rvs` calls and (2) ensure that all adaptive
              integrator steps in between user-defined steps use the same
              noise value (so that the dynamics are not constantly changing).
              Without this, the integrator fails to move forward in time.
              Therefore, a custom hash function is used to apply the same
              noise is necessary cases.
        
        Args:
            value (float): some value to be made noisier
            t (float): the current simulation time in seconds
            delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
        
        Returns:
            float: the value with some Gaussian noise applied
        """
        # randomly apply some noise, but use a cache so that each integrator-defined adaptive integration step
        # between user-defined integration steps uses the same noise value (otherwise the propagator fails)
        noise = self.noise_vals[int(t // delta_t) % len(self.noise_vals)]
        return value * (1 + noise)
