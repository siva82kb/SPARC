
"""
This module contains the a list of supporting classes and functions for the
SPARC smoothness analysis notebook.
"""

import numpy as np
import sys
from scipy import signal

# from smoothness import spectral_arclength
# from smoothness import log_dimensionless_jerk
from movements import gaussian_discrete_movement as gaussian_dm
from movements import generate_movement as gen_move
from smoothness import sparc
from smoothness import log_dimensionless_jerk as ldlj


# def dimensionless_jerk2(data, fs, data_type='speed'):
#     """
#     Calculates the smoothness metric for the given movement (position,
#     speed and acceleration) data using the dimensionless jerk.
#     """
#     # first enforce data into an numpy array.
#     data = np.array(data)

#     # calculate the scale factor and jerk.
#     data_peak = max(abs(data))
#     dt = 1. / fs
#     data_dur = len(data) * dt
#     if data_type == 'position':
#         jerk = np.diff(data, 3) / pow(dt, 3)
#         scale = 0.284444444444 * pow(data_dur, 5) / pow(data_peak, 2)
#     elif data_type == 'speed':
#         jerk = np.diff(data, 2) / pow(dt, 2)
#         scale = 1.0 * pow(data_dur, 3) / pow(data_peak, 2)
#     elif data_type == 'acceleration':
#         jerk = np.diff(data, 1) / pow(dt, 1)
#         scale = 9.48147733446 * pow(data_dur, 1) / pow(data_peak, 2)
#     else:
#         print ("Error! 'data_type' not recognized. It must be 'position'," +
#                " 'speed' or 'acceleration'")
#         return

#     # estimate dj
#     return - scale * sum(pow(jerk, 2)) * dt


# def log_dimensionless_jerk2(data, fs, data_type='speed'):
#     """
#     Calculates the smoothness metric for the movement data using the log
#     dimensionless jerk metric.
#     """
#     return -np.log(abs(dimensionless_jerk2(data, fs, data_type)))


def round_in(n, precision):
    """
    Rounds the given real number for the given precision to a number closer to
    zero.

    Parameters
    ----------
    n           : float
                  The number to be rounded in.
    precision   : integer
                  The precision to use for rounding the given number n.

    Returns
    -------
    sal      : float
               The rounded number.

    Notes
    -----

    Examples
    --------
    >>> round_in(-12.345, 0)
    -12.0
    >>> round_in(-12.345, 2)
    -12.4
    >>> round_in(12.345, 0)
    12.0
    >>> round_in(12.345, 1)
    12.3
    """
    scale = np.power(10.0, precision)
    return (np.ceil(scale * n) / scale if n <= 0
            else np.floor(scale * n) / scale)


def round_out(n, precision):
    """
    Rounds the given real number for the given precision to a number farther
    from zero.

    Parameters
    ----------
    n           : float
                  The number to be rounded in.
    precision   : integer
                  The precision to use for rounding the given number n.

    Returns
    -------
    sal      : float
               The rounded number.

    Notes
    -----

    Examples
    --------
    >>> round_out(-12.345, 0)
    -13.0
    >>> round_out(-12.332, 2)
    -12.34
    >>> round_out(12.345, 0)
    13.0
    >>> round_out(12.345, 1)
    12.4
    """
    scale = np.power(10.0, precision)
    return (np.floor(scale * n) / scale if n <= 0
            else np.ceil(scale * n) / scale)


# Time windows for selecting time series data.
class EstimationWindows(object):

    """
    Base class for windows to select data.
    """

    def __init__(self, amp, t_dur, t_loc, ts):
        t_dur = float(t_dur)
        t_loc = float(t_loc)
        self._param = {'amp': amp, 'dur': t_dur, 'loc': t_loc, 'ts': ts}
        self.edges = (t_loc - 0.5 * t_dur, t_loc + 0.5 * t_dur)


class RectangularWindow(EstimationWindows):

    """
    Implements a rectangular window for selecting data.
    """

    def get_window(self, time=None):
        """
        Returns the window values for the given time (list).
        """
        time = time if (time is not None) else np.arange(self.edges[0],
                                                         self.edges[1],
                                                         self._param['ts'])
        # check if the time is a scalar.
        if type(time) in (int, float):
            return time, 1. if (time >= self.edges[0] and
                                time <= self.edges[1]) else 0.
        else:
            win = np.array([t >= self.edges[0] and t <= self.edges[1]
                            for t in time])
            win = self._param['amp'] * (win.astype(float))

        return time, win


class GaussianWindow(EstimationWindows):

    """
    Implements a Gaussian window for selecting data.
    """

    def get_window(self, time=None):
        """
        Returns the window values for the given time (list).
        """
        time = time if (time is not None) else np.arange(self.edges[0],
                                                         self.edges[1],
                                                         self._param['ts'])

        amp = self._param['amp']
        tdur = self._param['dur']
        tloc = self._param['loc']

        return time, gaussian_dm(amp, tdur, tloc, time)


def generate_ideal_movements(param, ts=0.001):
    """Generates a set of ideal movements using the given parameters.
    """
    sys.stdout.write('.')
    moves = []
    for i in xrange(param['N_m']):
        # Generate ideal movement
        _, _move, _ = gen_move(Ns=param['Ns'][i],
                               amp=param['amps'][i],
                               dT=param['dTs'][i],
                               T=param['Ts'][i],
                               ts=ts)
        # Append to ideal movements
        moves.append(_move)
    return np.array(moves)


def generate_noisy_movements(ideal_moves, param):
    """Generates noisy versions of the given set of ideal movements.
    """
    sys.stdout.write('.')
    noisy_moves = []
    for i in xrange(param['N_m']):
        # Average power in the movement data.
        _var = np.mean(np.power(ideal_moves[i], 2))
        # go through the different SNR
        _noisy_move = []
        for j, _snr in enumerate(param['snr']):
            _n_move = []
            for k in xrange(param['N_n']):
                _tmp = (ideal_moves[i] +
                        (np.sqrt(_var / _snr) *
                         np.random.randn(len(ideal_moves[i]))))
                _n_move.append(_tmp)
            _noisy_move.append(_n_move)
        noisy_moves.append(_noisy_move)
    return np.array(noisy_moves)


def filter_parameters(fs):
    """Generates and returns the filter parameters for filtering the noisy
    movement data.
    """
    filt_params = {'N': [2, 8],
                   'fc': [10., 15., 20.],
                   'b': [],
                   'a': []}
    # Generate Buuterworth filter coefficients
    temp = np.array([[signal.butter(filt_params['N'][i],
                                    filt_params['fc'][j] / (0.5 * fs))
                      for j in xrange(len(filt_params['fc']))]
                     for i in xrange(len(filt_params['N']))])
    filt_params['b'] = temp[:, :, 0]
    filt_params['a'] = temp[:, :, 1]
    return filt_params


def filter_noisy_movements(noisy_moves, param, filt_param):
    """Filters the noisy data set with the given filter parameters.
    """
    sys.stdout.write('.')
    return np.array(
        [[[[[signal.filtfilt(filt_param['b'][l, m],
                             filt_param['a'][l, m],
                             noisy_moves[i, j, k],
                             padlen=len(noisy_moves[i, j, k]) - 10)
             for m in xrange(len(filt_param['fc']))]
            for l in xrange(len(filt_param['N']))]
           for k in xrange(param['N_n'])]
          for j in xrange(len(param['snr']))]
         for i in xrange(param['N_m'])])


def est_smooth_ideal(moves, ts, stype='sparc'):
    """Estimates the smoothness of the ideal movements.
    """
    sys.stdout.write('.')
    if stype == 'sparc':
        return np.array(
            [sparc(_m, fs=1 / ts, padlevel=4, fc=10., amp_th=0.05)[0]
             for _m in moves])
    else:
        return np.array([ldlj(_m, fs=1 / ts) for _m in moves])


def est_smooth_noisy(moves, ts, param, stype='sparc'):
    """Estimates the smoothness of noisy movements.
    """
    sys.stdout.write('.')
    if stype == 'sparc':
        return np.array(
            [[[sparc(moves[i, j, k], fs=1 / ts,
                     padlevel=4, fc=10., amp_th=0.05)[0]
               for k in xrange(param['N_n'])]
              for j in xrange(len(param['snr']))]
             for i in xrange(param['N_m'])])
    else:
        return np.array(
            [[[ldlj(moves[i, j, k], ts)
               for k in xrange(param['N_n'])]
              for j in xrange(len(param['snr']))]
             for i in xrange(param['N_m'])])


def est_smooth_filt(moves, ts, param, filt_param, stype='sparc'):
    """Estimate the smoothness of filtered movements.
    """
    sys.stdout.write('.')
    if stype == 'sparc':
        return np.array(
            [[[[[sparc(moves[i, j, k, l, m], fs=1 / ts,
                       padlevel=4, fc=10., amp_th=0.05)[0]
                 for m in xrange(len(filt_param['fc']))]
                for l in xrange(len(filt_param['N']))]
               for k in xrange(param['N_n'])]
              for j in xrange(len(param['snr']))]
             for i in xrange(param['N_m'])])
    else:
        return np.array(
            [[[[[ldlj(moves[i, j, k, l, m], ts)
                 for m in xrange(len(filt_param['fc']))]
                for l in xrange(len(filt_param['N']))]
               for k in xrange(param['N_n'])]
              for j in xrange(len(param['snr']))]
             for i in xrange(param['N_m'])])


def est_change_noisy(s_noisy, s_ideal, param):
    """Estimates the percentage change in smoothness because of noisy.
    """
    return np.array(
        [[[100.0 * (s_noisy[i, j, k] - s_ideal[i]) / s_ideal[i]
           for k in xrange(param['N_n'])]
          for j in xrange(len(param['snr']))]
         for i in xrange(param['N_m'])])


def est_change_noisy_wrt_range(s_noisy, s_ideal, param):
    """Estimates the percentage change in smoothness because of noise
    with respect to the range of smoothness values.
    """
    _range = np.abs(np.max(s_ideal) - np.min(s_ideal))
    return np.array(
        [[[100.0 * (s_noisy[i, j, k] - s_ideal[i]) / _range
           for k in xrange(param['N_n'])]
          for j in xrange(len(param['snr']))]
         for i in xrange(param['N_m'])])


def est_change_filt_wrt_range(s_filt, s_ideal, param, filt_param):
    """Estimates the percentage change in smoothness in the
    filtered noisy movement data with respect to the range of
    smoothness values.
    """
    _range = np.abs(np.max(s_ideal) - np.min(s_ideal))
    return np.array(
        [[[[[100.0 * (s_filt[i, j, k, l, m] - s_ideal[i]) / _range
             for m in xrange(len(filt_param['fc']))]
            for l in xrange(len(filt_param['N']))]
           for k in xrange(param['N_n'])]
          for j in xrange(len(param['snr']))]
         for i in xrange(param['N_m'])])


def est_change_filt(s_filt, s_ideal, param, filt_param):
    """Estimates the percentage change in smoothness in the
    filtered noisy movement dat.
    """
    return np.array(
        [[[[[100.0 * (s_filt[i, j, k, l, m] - s_ideal[i]) / s_ideal[i]
             for m in xrange(len(filt_param['fc']))]
            for l in xrange(len(filt_param['N']))]
           for k in xrange(param['N_n'])]
          for j in xrange(len(param['snr']))]
         for i in xrange(param['N_m'])])


def genenrate_different_scale_move(param, ts=0.001):
    """Generates movements of two different scales.
    """
    return np.array([[gen_move(Ns=param['Ns'][i],
                               amp=(1.0 / _s) * param['amps'][i],
                               dT=_s * param['dTs'][i],
                               T=_s * param['Ts'][i],
                               ts=ts)[1]
                      for i in xrange(param['N_m'])]
                     for _s in param['scales']])


def add_noise(moves, noise_p, param):
    """Adds noise to the movement data.
    """
    return np.array(
        [[moves[i, j] + np.sqrt(noise_p) * np.random.randn(len(moves[i, j]))
          for j in xrange(param['N_m'])]
         for i, _s in enumerate(param['scales'])])


def filter_movements(moves, fs, param):
    """Filters the noisy movement data.
    """
    b, a = signal.butter(8.0, 10.0 / (0.5 * fs))
    return np.array([[signal.filtfilt(b, a, moves[i, j],
                                      padlen=len(moves[i, j]) - 10)
                      for j in xrange(param['N_m'])]
                     for i, _s in enumerate(param['scales'])])


def estimate_smoothness(moves, ts, param, stype="sparc"):
    """Estimates the smoothness of the movements of two different scales.
    """
    if stype == 'sparc':
        return np.array(
            [[sparc(moves[i, j], fs=1 / ts, padlevel=4, fc=10., amp_th=0.05)[0]
              for j in xrange(param['N_m'])]
             for i in xrange(len(param['scales']))])
    else:
        return np.array(
            [[ldlj(moves[i, j], ts)
              for j in xrange(param['N_m'])]
             for i in xrange(len(param['scales']))])
