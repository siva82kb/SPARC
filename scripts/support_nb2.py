
"""
This module contains the a list of supporting classes and functions for the
notebook investigating smoothness estimates directly from different types of
signals. (Notebook name: sparc_ldlj_demo_for_other_data_types.ipynb)
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

from movements import generate_movement as gen_move
from smoothness import sparc
from smoothness import log_dimensionless_jerk2 as ldlj2
# from movements import gaussian_discrete_movement as gaussian_dm


out_dir = "img_nb2"


def generate_simulated_movements(Ns, dT, Ts, ts, move_type):
    """Generates a set of movements with different submovement numbers
    and intervals."""
    moves = []
    for ni, n in enumerate(Ns):
        _temp = []
        for dt in dT:
            sys.stdout.write('\rNs: {0}, dT: {1}'.format(n, dt))
            t, m, _ = gen_move(n, [1. / n] * n, [dt] * (n - 1), [Ts] * n,
                               ts=ts, move_type=move_type)
            _temp.append(m)
        moves.append(_temp)
    return moves


def smoothness_summary_plot(Ns, dT, m_types, smooth_vals):
    """Summary of the smoothness measures as a function of the number of
    submovements and the inter-submovement interval.
    """
    figs = []
    for _n, _type in enumerate(m_types):
        fig = plt.figure(figsize=(10, 6))
        # vs. inter-submovement interval
        ax = fig.add_subplot(2, 2, 1)
        for i, _ in enumerate(Ns):
            plt.plot(dT, smooth_vals[_type]['sparc'][i, :])
        ax.set_title('[{0}] SPARC vs. $\Delta T$'.format(_type))
        ax = fig.add_subplot(2, 2, 2)
        for i, _ in enumerate(dT):
            if i % 10 == 0:
                plt.plot(Ns, smooth_vals[_type]['sparc'][:, i])
        ax.set_title('[{0}] SPARC vs. Ns'.format(_type))
        ax = fig.add_subplot(2, 2, 3)
        for i, _ in enumerate(Ns):
            plt.plot(dT, smooth_vals[_type]['ldlj'][i, :])
        ax.set_title('[{0}] LDLJ vs. $\Delta T$'.format(_type))
        ax.set_xlabel('Time (sec)')
        ax = fig.add_subplot(2, 2, 4)
        for i, _ in enumerate(dT):
            if i % 10 == 0:
                plt.plot(Ns, smooth_vals[_type]['ldlj'][:, i])
        ax.set_title('[{0}] LDLJ vs. Ns'.format(_type))
        ax.set_xlabel('No. of submovements')
        plt.tight_layout()
        # save figure
        try:
            os.makedirs(out_dir)
        except:
            pass
        _fname = "smooth_summary_{0}".format(_type)
        fig.savefig(os.path.join(out_dir, "{0}.png".format(_fname)),
                    format='png', dpi=600)
        fig.savefig(os.path.join(out_dir, "{0}.svg".format(_fname)),
                    format='svg', dpi=600)
        # tikz_save(os.path.join(out_dir, "{0}.tex".format(_fname)))
        figs.append(fig)
    return figs


def estimate_smoothness_values(moves, Ns, dT, ts, m_types):
    # smoothness estimates
    # smooth_vals = {'speed': {'sparc': [], 'ldlj': []},
    #                'accl': {'sparc': [], 'ldlj': []},
    #                'jerk': {'sparc': [], 'ldlj': []}}
    _str = '\rType: {0}, Ns: {1}, dT: {2}'
    smooth_vals = {}
    for _n, _type in enumerate(m_types):
        _temp1 = np.zeros((len(Ns), len(dT)))
        _temp2 = np.zeros((len(Ns), len(dT)))
        for i in xrange(len(Ns)):
            _tmp1 = []
            _tmp2 = []
            for j in xrange(len(dT)):
                sys.stdout.write(_str.format(_type, Ns[i], dT[j]))
                m = np.diff(moves[i][j], n=_n) / np.power(ts, _n)
                _tmp1.append(sparc(np.abs(m), fs=1 / ts)[0])
                _tmp2.append(ldlj2(m, 1 / ts, _type))
            _temp1[i, :] = _tmp1
            _temp2[i, :] = _tmp2
        smooth_vals[_type] = {'sparc': _temp1, 'ldlj': _temp2}
    return smooth_vals


def get_time_and_move(moves, row, col, ts, m_type):
    if m_type == 'speed':
        _m = moves[row][col]
    elif m_type == 'accl':
        _m = np.diff(moves[row][col]) / ts
    else:
        _m = np.diff(moves[row][col], 2) / np.power(ts, 2)
    _t = np.arange(0, len(_m) * ts, ts)
    return _t, _m


def movement_profile_plots(moves, Ns, dT, delT, ts, m_types):
    dTs = [0.25, 0.5, 1.0, 1.8]
    dTinx = [int(round(_dt / delT) - 1) for _dt in dTs]

    # Plot of the speed, acceleration adn jerk profiles for three
    # different inter-submovement interval values.
    Nr, Nc = len(m_types), len(dTinx)
    fig = plt.figure(figsize=(12, 7))
    for row, _type in enumerate(m_types):
        for col, _dtinx in enumerate(dTinx):
            _inx = Nc * row + col + 1
            ax = fig.add_subplot(Nr, Nc, _inx)
            _t, _m = get_time_and_move(moves, 4, dTinx[col], ts, _type)
            plt.plot(_t, _m)
            _str = "[{0}] $\Delta$T = {1}, Ns = {2}"
            ax.set_title(_str.format(_type, dTs[col], Ns[4]))
            if row == len(m_types) - 1:
                ax.set_xlabel('Time (s)')
    plt.tight_layout()
    _fname = "movement_profiles"
    fig.savefig(os.path.join(out_dir, "{0}.png".format(_fname)),
                format='png', dpi=600)
    fig.savefig(os.path.join(out_dir, "{0}.svg".format(_fname)),
                format='svg', dpi=600)


def _dlj(movement, fs, data_type='speed'):
    """
    Returns the different factors of the Dimensionless jerk metric.
    """
    # first ensure the movement type is valid.
    if data_type in ('speed', 'accl', 'jerk'):
        # first enforce data into an numpy array.
        movement = np.array(movement)

        # calculate the scale factor and jerk.
        movement_peak = max(abs(movement))
        dt = 1. / fs
        movement_dur = len(movement) * dt
        # get scaling factor:
        _p = {'speed': 3,
              'accl': 1,
              'jerk': -1}
        p = _p[data_type]
        dur, amp = np.power(movement_dur, p), 1 / np.power(movement_peak, 2)

        # estimate jerk
        if data_type == 'speed':
            jerk = np.diff(movement, 2) / np.power(dt, 2)
        elif data_type == 'accl':
            jerk = np.diff(movement, 1) / np.power(dt, 1)
        else:
            jerk = movement

        # estimate dj
        return amp, dur, sum(np.power(jerk, 2)) * dt
    else:
        raise ValueError('\n'.join(("The argument data_type must be either",
                                    "'speed', 'accl' or 'jerk'.")))


def _ldlj(movement, fs, data_type='speed'):
    """
    Returns the different factors of the log dimensionless jerk metric.
    """
    _amp, _dur, _jerk = _dlj(movement, fs, data_type)
    return np.log(_amp), np.log(_dur), np.log(_jerk)


def ldlj_factors(moves, Ns, dT, ts, m_types):
    # Different factors of the LDLJ measure as a function of
    # inter-submovement interval.
    amp = np.zeros((len(m_types), len(Ns), len(dT)))
    dur = np.zeros((len(m_types), len(Ns), len(dT)))
    intgnd = np.zeros((len(m_types), len(Ns), len(dT)))

    for i1 in xrange(len(m_types)):
        for i2 in xrange(len(Ns)):
            for i3 in xrange(len(dT)):
                if m_types[i1] == 'speed':
                    _m = moves[i2][i3]
                elif m_types[i1] == 'accl':
                    _m = np.diff(moves[i2][i3]) / ts
                else:
                    _m = np.diff(moves[i2][i3], 2) / np.power(ts, 2)
                _a, _d, _i = _ldlj(_m, 1. / ts, m_types[i1])
                amp[i1, i2, i3] = _a
                dur[i1, i2, i3] = _d
                intgnd[i1, i2, i3] = _i

    # Plot the different factors for the different movement types.
    fig = plt.figure(figsize=(15, 3))
    dur_pow = [3, 1, -1]
    for inx, _type in enumerate(m_types):
        ax = fig.add_subplot(1, 3, inx + 1)
        plt.plot(dT, -amp[inx, 4, :], label="$1/x_p^2$", lw=1)
        plt.plot(dT, -dur[inx, 4, :],
                 label="$\Delta T^{{{0}}}$".format(dur_pow[inx]), lw=1)
        plt.plot(dT, -intgnd[inx, 4, :],
                 label="$\int_{0}^{T}\|j(t)\|^2dt$", lw=1)
        plt.plot(dT, -amp[inx, 4, :] - dur[inx, 4, :] - intgnd[inx, 4, :],
                 label="LDLJ")
        ax.set_xlim(0, 3.5)
        ax.set_title(m_types[inx])
        ax.set_xlabel("Inter-submovement interval (s)")
        plt.legend()
    _fname = "ldlj_scales"
    fig.savefig(os.path.join(out_dir, "{0}.png".format(_fname)),
                format='png', dpi=600)
    fig.savefig(os.path.join(out_dir, "{0}.svg".format(_fname)),
                format='svg', dpi=600)
