# -*- coding: utf-8 -*-
"""Simulation module for attitude determination and control system.

This module forms the core of the simulation engine and utilizes the classes
and functions written elsewhere to model the system and perform numerical
integration.
"""
import torch
import numpy as np
from scipy.integrate import ode, RK45, odeint
from adcs_simulation.kinematics import quaternion_derivative
from adcs_simulation.dynamics import angular_velocity_derivative
from adcs_simulation.math_utils import normalize
from adcs_simulation.errors import calculate_attitude_error, calculate_attitude_rate_error
from adcs_simulation.euler_solver import SDE, integrate
import matplotlib.pyplot as plt

def simulate_adcs(satellite,
                  nominal_state_func,
                  perturbations_func,
                  position_velocity_func,
                  start_time=0,
                  delta_t=1,
                  stop_time=6000,
                  rtol=(1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-6, 1e-6,
                      1e-6),
                  atol=(1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-8, 1e-8,
                      1e-8),
                  verbose=False):
    """Simulates an attitude determination and control system over a period of time
    
    Args:
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        nominal_state_func (function): the function that should compute the
            nominal attitude (in DCM form) and angular velocity; its header
            must be (t)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (satellite)
        position_velocity_func (function): the function that should compute
            the position and velocity; its header must be (t)
        start_time (float, optional): Defaults to 0. The start time of the
            simulation in seconds
        delta_t (float, optional): Defaults to 1. The time between user-defined
            integrator steps (not the internal/adaptive integrator steps) in
            seconds
        stop_time (float, optional): Defaults to 6000. The end time of the
            simulation in seconds
        rtol (tuple): Relative tolerances for the solver, corresponding to
            the satellite state vector (10x1)
        rtol (tuple): Absolute tolerances for the solver, corresponding to
            the satellite state vector (10x1)
        verbose (bool, optional). Defaults to False. Whether or not to print
            integrator output to the console while running.
    
    Returns:
        dict: a dictionary of simulation results. Each value is an NxM numpy
            ndarray where N is the number of time steps taken and M is the
            size of the data being represented at that time (M=1 for time, 
            3 for angular velocity, 4 for quaternion, etc.)
            Contains:
                - times (numpy ndarray): the times of all associated data
                - q_actual (numpy ndarray): actual quaternion
                - w_actual (numpy ndarray): actual angular velocity
                - w_rxwls (numpy ndarray): angular velocity of the reaction
                    wheels
                - DCM_estimated (numpy ndarray): estimated DCM
                - w_estimated (numpy ndarray): estimated angular velocity
                - DCM_desired (numpy ndarray): desired DCM
                - w_desired (numpy ndarray): desired angular velocity
                - attitude_err (numpy ndarray): attitude error
                - attitude_rate_err (numpy ndarray): attitude rate error
                - M_ctrl (numpy ndarray): control torque
                - M_applied (numpy ndarray): applied control torque
                - w_dot_rxwls (numpy ndarray): angular acceleration of
                    reaction wheels
                - M_perturb (numpy ndarray): sum of perturbation torques
                - positions (numpy ndarray): inertial positions
                - velocities (numpy ndarray): inertial velocities

    """
    try:
        init_state = [*satellite.q, *satellite.w, *satellite.actuators.w_rxwls, *np.zeros(6)]
    except AttributeError:
        init_state = [*satellite.q, *satellite.w, 0, 0, 0, *np.zeros(6)]
    solver = SDE(f_and_g_func,
                    satellite,
                    nominal_state_func,
                    perturbations_func,
                    position_velocity_func,
                    delta_t)
    ts = np.linspace(0, stop_time, int(stop_time / delta_t))
    results = integrate(solver, init_state, ts)
    plt.figure()
    for i in range(10,13):
        data = [x[0][i].item() for x in results]
        plt.plot(data)
    plt.title('Attitude error')
    plt.figure()
    for i in range(13,16):
        data = [x[0][i].item() for x in results]
        plt.plot(data)
    plt.title('Attitude rate error')
    plt.show()

def f_and_g_func(time, y, satellite, nominal_state_func, perturbations_func,
                     position_velocity_func, delta_t):
    """Computes the derivative of the spacecraft state
    
    Args:
        t (float): the time (in seconds)
        x (numpy ndarray): the state (10x1) where the elements are:
            [0, 1, 2, 3]: the quaternion describing the spacecraft attitude
            [4, 5, 6]: the angular velocity of the spacecraft
            [7, 8, 9]: the angular velocities of the reaction wheels
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        nominal_state_func (function): the function that should compute the
            nominal attitude (in DCM form) and angular velocity; its header
            must be (t)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (satellite)
        position_velocity_func (function): the function that should compute
            the position and velocity; its header must be (t)
        delta_t: Timestep of the analysis (seconds)
   
    Returns:
        numpy ndarray: the derivative of the state (10x1) with respect to time
    """
    # TODO: Use tensors deeper in the program to speed up the processing further
    t = time.item()
    print(t)
    x = np.array(y[0].tolist())
    r, v = position_velocity_func(t)
    satellite.q = normalize(x[0:4])
    satellite.w = x[4:7]
    satellite.r = r
    satellite.v = v
    # only set if the satellite has actuators
    try:
        satellite.actuators.w_rxwls = x[7:10]
    except AttributeError:
        pass
    M_applied, w_dot_rxwls, log = simulate_estimation_and_control(
        t, satellite, nominal_state_func, delta_t)
    
#    M_applied_noise, w_dot_rxwls_noise = satellite.actuators.apply_noise_torques(satellite.w, t, delta_t)
    
    # calculate the perturbing torques on the satellite
    M_perturb = perturbations_func(satellite)

    # fx are the derivatives for the "drift" part of SDE
    # so normal movement
    fx = np.zeros(len(x))

    # gx are the derivatives for the "diffusion" part of SDE
    # so the noise part
    # TODO: Implement noise separately
    gx = np.zeros(len(x))

    # This is sunstorm attitude control error. Not used properly here but something to start with
    angular_error = 2e-4
 #   angular_speed_error_samples = np.random.randn(1, 3) * angular_error
    
    fx[0:4] = quaternion_derivative(satellite.q, satellite.w)
    fx[4:7] = angular_velocity_derivative(satellite.J, satellite.w,
                                          [M_applied, M_perturb])
#    gx[4:7] = angular_velocity_derivative(satellite.J, satellite.w, [M_applied_noise]) + angular_speed_error_samples
    fx[7:10] = w_dot_rxwls
    fx[10:13] = log['attitude_err']
    fx[13:16] = log['attitude_rate_err']
 #   gx[7:10] = w_dot_rxwls_noise
    return torch.tensor([fx], dtype=torch.float64),torch.tensor([gx], dtype=torch.float64)

def simulate_estimation_and_control(t,
                                    satellite,
                                    nominal_state_func,
                                    delta_t,
                                    log=True):
    """Simulates attitude estimation and control for derivatives calculation
    
    Args:
        t (float): the time (in seconds)
        satellite (Spacecraft): the Spacecraft object that represents the
            satellite being modeled
        nominal_state_func (function): the function that should compute the
            nominal attitude (in DCM form) and angular velocity; its header
            must be (t)
        perturbations_func (function): the function that should compute the
            perturbation torques (N * m); its header must be (t, q, w)
        delta_t (float): the time between user-defined integrator steps
                (not the internal/adaptive integrator steps) in seconds
    
    Returns:
        numpy ndarray: the control moment (3x1) as actually applied on
                the reaction wheels (the input control torque with some
                Gaussian noise applied) (N * m)
        numpy ndarray: the angular acceleration of the 3 reaction wheels
            applied to achieve the applied torque (rad/s^2)
        dict: a dictionary containing results logged for this simulation step;
            Contains:
                - DCM_estimated (numpy ndarray): estimated DCM
                - w_estimated (numpy ndarray): estimated angular velocity
                - DCM_desired (numpy ndarray): desired DCM
                - w_desired (numpy ndarray): desired angular velocity
                - attitude_err (numpy ndarray): attitude error
                - attitude_rate_err (numpy ndarray): attitude rate error
                - M_ctrl (numpy ndarray): control torque
                - M_applied (numpy ndarray): applied control torque
                - w_dot_rxwls (numpy ndarray): angular acceleration of
                    reaction wheels
    """
    # get an attitude and angular velocity estimate from the sensors
    w_estimated = satellite.estimate_angular_velocity(t, delta_t)
    DCM_estimated = satellite.estimate_attitude(t, delta_t)

    # compute the desired attitude and angular velocity
    DCM_desired, w_desired = nominal_state_func(t)

    # calculate the errors between your desired and estimated state
    attitude_err = calculate_attitude_error(DCM_desired, DCM_estimated)
    attitude_rate_err = calculate_attitude_rate_error(w_desired, w_estimated,
                                                      attitude_err)

    # determine the control torques necessary to change state
    M_ctrl = satellite.calculate_control_torques(attitude_err,
                                                 attitude_rate_err)
    # use actuators to apply the control torques
    M_applied, w_dot_rxwls = satellite.apply_control_torques(
        M_ctrl, t, delta_t)

    if log:
        logged_results = {
            "DCM_estimated": DCM_estimated,
            "w_estimated": w_estimated,
            "DCM_desired": DCM_desired,
            "w_desired": w_desired,
            "attitude_err": attitude_err,
            "attitude_rate_err": attitude_rate_err,
            "M_ctrl": M_ctrl,
            "M_applied": M_applied,
            "w_dot_rxwls": w_dot_rxwls
        }
    else:
        logged_results = None

    return M_applied, w_dot_rxwls, logged_results
