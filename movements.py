"""
movements.py

This module provides a set of functions to model different types of motion
profiles over a specified duration. These functions return the position
at a given time `t` according to the chosen motion law.

Available motion profiles:
- linear_motion: Linear position from start to end.
- quadratic_motion: Quadratically increasing position (x(t) ~ t^2).
- log_motion: Logarithmic position, fast initial growth then slows down.
- exp_motion: Exponential position, slow initial growth then accelerates.
- sinusoidal_motion: Sinusoidal position oscillating between a min and max.

Each function ensures that the position remains within the defined bounds
and avoids zero values at t=0 for numerical stability.

Function signatures:
    linear_motion(t, total_time=1, total_distance=0.1)
    quadratic_motion(t, total_time=1, total_distance=0.1)
    log_motion(t, total_time=1, total_distance=0.1)
    exp_motion(t, total_time=1, total_distance=0.1)
    sinusoidal_motion(t, total_time=1.0, min_pos=0.001, max_pos=0.01)

Parameters (common to most functions):
    t : float
        Current time in seconds.
    total_time : float
        Total duration of the motion.
    total_distance : float
        Total distance to cover (for linear, quadratic, log, and exp motions).
    min_pos : float
        Minimum position (for sinusoidal motion).
    max_pos : float
        Maximum position (for sinusoidal motion).

Returns:
    float
        The position at time t in meters.
"""

def linear_motion(t, total_time=1, total_distance=0.1):
    if t <= 0:
        return 0.001
    elif t >= total_time:
        return total_distance
    else:
        return (total_distance / total_time) * t
    
def quadratic_motion(t, total_time=1, total_distance=0.1):
    if t <= 0:
        return 0.001
    elif t >= total_time:
        return total_distance
    else:
        return 0.001 + (total_distance - 0.001) * (t / total_time)**2

def log_motion(t, total_time=1, total_distance=0.1):
    if t <= 0:
        return 0.001
    elif t >= total_time:
        return total_distance
    else:
        # Normalisation : log(1 + (e-1) * (t/total_time)) varie de 0 à 1
        progress = np.log(1 + (np.e - 1) * (t / total_time)) / 1.0
        return 0.001 + (total_distance - 0.001) * progress

def exp_motion(t, total_time=1, total_distance=0.1):
    if t <= 0:
        return 0.001
    elif t >= total_time:
        return total_distance
    else:
        # Normalisation : (exp(k*x) - 1)/(exp(k) - 1) varie de 0 à 1
        k = 5.0  # facteur de "raideur" de l'exponentielle
        progress = (np.exp(k * (t / total_time)) - 1) / (np.exp(k) - 1)
        return 0.001 + (total_distance - 0.001) * progress

def sinusoidal_motion(t, total_time=1.0, min_pos=0.001, max_pos=0.01):
    if t <= 0:
        return min_pos
    elif t >= total_time:
        return max_pos
    else:
        # Amplitude et décalage
        amplitude = (max_pos - min_pos) / 2.0
        offset = (max_pos + min_pos) / 2.0

        # Demi-période de sinusoïde de -pi/2 à +pi/2
        phase = (t / total_time) * np.pi - np.pi / 2

        return offset + amplitude * np.sin(phase)