"""Module for generating plots for the paper."""

import numpy as np
import matplotlib.pyplot as plt

from smoothness import sparc
from smoothness import log_dimensionless_jerk


def sine_rhythmic_movement(T_m, T_r, T_t, ts, skill=1):
    # time
    t = np.arange(0, T_t, ts)

    # Total number of movements
    N = int(np.floor(T_t/(2*T_m + 2*T_r)))

    t, _movement, _vel, _speed, _move_tag, _rest_tag = sine_rhythmic_movement_by_number(T_m, T_r, N, ts, skill)

    return t, _movement, _vel, _speed, _move_tag, _rest_tag, N


def sine_rhythmic_movement_by_number(T_m, T_r, N, ts, skill=1):
    # One period of movement and rest
    _temp = np.concatenate(
        (0.5 - 0.5 * np.cos(np.pi * np.arange(0, T_m, ts) / T_m),
         np.ones(T_r/ts),
         0.5 + 0.5 * np.cos(np.pi * np.arange(0, T_m, ts) / T_m),
         np.zeros(T_r/ts)))
    _movement = np.tile(_temp, N)

    # tag for movements.
    # P -> Q is +1
    # Q -> P is -1
    # Everything else is 0.
    _temp = np.concatenate(
        (np.ones(T_m/ts),
         np.zeros(T_r/ts),
         -1.0 * np.ones(T_m/ts),
         np.zeros(T_r/ts)))
    _move_tag = np.tile(_temp, N)

    # tag for movements.
    # @P 1
    # @Q -1
    # Everything else is 0.
    _temp = np.concatenate(
        (np.zeros(T_m/ts),
         -1 * np.ones(T_r/ts),
         np.zeros(T_m/ts),
         np.ones(T_r/ts)))
    _rest_tag = np.tile(_temp, N)

    # If skill is less than 1, then add some random high frequency movements.
    for n in xrange(N):
        _amp = (1 - skill) * np.random.rand(2)
        _freq = np.random.randint(3, 6, 2)
        _temp = np.concatenate(
            (_amp[0] * np.sin(_freq[0] * np.pi * np.arange(0, T_m, ts) / T_m),
             np.zeros(T_r/ts),
             _amp[1] * np.sin(_freq[1] * np.pi * np.arange(0, T_m, ts) / T_m),
             np.zeros(T_r/ts)))
        if n == 0:
            _noise = _temp
        else:
            _noise = np.append(_noise, _temp)

    # movement with noise
    _movement += _noise

    # velcoity and speed of the movement.
    _vel = np.zeros(len(_movement))
    _vel[1:] = np.diff(_movement) / ts
    _speed = np.abs(_vel)
    # time
    t = np.arange(0, ts * len(_movement), ts)
    return t, _movement, _vel, _speed, _move_tag, _rest_tag


def changing_sine_rhythmic_movement(T_t, ts, skill=1):
    t = np.arange(0, T_t, ts)

    # movement and rest time trends.
    _tr_trend = 0.52 * np.power(t - 15.0, 2.0) / 225.
    _tm_trend = 0.4 * np.power(t - 15.0, 2.0) / 225. + 0.6

    # First movement
    _, m, _, _, _, _, _ = sine_rhythmic_movement(T_m=_tm_trend[0],
                                                 T_r=_tr_trend[0],
                                                 T_t=2 * (_tm_trend[0] +
                                                          _tr_trend[0]),
                                                 ts=0.01)
    # create successive movements and append them to m
    _t = len(m) * ts
    while _t <= t[-1]:
        _inx = np.nonzero(np.abs(t - _t) < 0.001)[0][0]
        _T_m = _tm_trend[_inx]
        _T_r = _tr_trend[_inx]
        _T_t = 2 * (_T_m + _T_r)
        _, _temp, _, _, _, _, _ = sine_rhythmic_movement(T_m=_T_m,
                                                         T_r=_T_r,
                                                         T_t=_T_t,
                                                         ts=0.01)
        _t += len(_temp) * ts
        m = np.append(m, _temp)

    # Time for movement, velocity and speed
    t = np.arange(0, len(m) * ts, ts)
    v = np.zeros(len(m))
    v[1:] = np.diff(m) / ts
    s = np.abs(v)

    return t, m, v, s


def plot_different_tasks(t1, m1, s1, t2, m2, s2, t3, m3, s3):
    fig = plt.figure(figsize=(10, 7))
    plt.subplot(311)
    plt.plot(t1, m1, '0.0', lw=2, label="Position")
    plt.plot(t1, 0.75 * s1 / np.max(s1), '0.4', lw=1, label="Speed")
    plt.ylabel('Position', fontsize=18)
    plt.xticks([], fontsize=18)
    plt.yticks([0, 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.5)
    plt.title('M1: Rhythmic movement with some dwell-time', fontsize=20)
    plt.text(0.5, 1.1, "(A)", fontsize=20)
    plt.legend(ncol=2, fontsize=20)

    plt.subplot(312)
    plt.plot(t2, m2, '0.0', lw=2, label="Position")
    plt.plot(t2, 0.75 * s2 / np.max(s2), '0.4', lw=1, label="Speed")
    plt.ylabel('Position', fontsize=18)
    plt.xticks([], fontsize=18)
    plt.yticks([0, 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.5)
    plt.title('M2: Rhythmic movement with zero dwell-time', fontsize=20)
    plt.text(0.5, 1.1, "(B)", fontsize=20)
    plt.legend(ncol=2, fontsize=20)

    plt.subplot(313)
    plt.plot(t3, m3, '0.0', lw=2, label="Position")
    plt.plot(t3, 0.75 * s3 / np.max(s3), '0.4', lw=1, label="Speed")
    plt.xlabel('Time (sec)', fontsize=20)
    plt.ylabel('Position', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks([0, 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.5)
    plt.xlim(0, 30.)
    plt.title('M3: Rhythmic movement with changing speed and dwell-time',
              fontsize=20)
    plt.text(0.5, 1.1, "(C)", fontsize=20)
    plt.legend(ncol=2, fontsize=20)

    plt.tight_layout()
    return fig


def plot_skilled_unskilled_tasks(t1, m1, t2, m2):
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(t1, m1, '0.0', lw=2, label="Position")
    plt.ylabel('Position', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks([0., 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.3)
    plt.title('M1a: Rhythmic movement performed by a skilled subject',
              fontsize=20)
    plt.text(0.5, 1.1, "(A)", fontsize=20)

    plt.subplot(212)
    plt.plot(t2, m2, '0.0', lw=2, label="Position")
    plt.xlabel('Time (sec)', fontsize=20)
    plt.ylabel('Position', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks([0., 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.3)
    plt.title('M1b: Rhythmic movement performed by a novice subject',
              fontsize=20)
    plt.text(0.5, 1.1, "(B)", fontsize=20)

    plt.tight_layout()
    return fig


def plot_three_simple_tasks(t1, m1, t2, m2, t3, m3):
    t_max = np.max([t1[-1], t2[-1], t3[-1]])
    fig = plt.figure(figsize=(10, 7))
    plt.subplot(311)
    plt.plot(t1, m1, '0.0', lw=2, label="Position")
    plt.ylabel('Position', fontsize=20)
    plt.xticks([], fontsize=18)
    plt.yticks([0., 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.3)
    plt.xlim(0, np.max([t1[-1], t2[-1], t3[-1]]))
    plt.title('Ma: Rhythmic movement (expert 1)',
              fontsize=20)
    plt.text(0.5, 1.1, "(A)", fontsize=20)

    plt.subplot(312)
    plt.plot(t2, m2, '0.0', lw=2, label="Position")
    plt.ylabel('Position', fontsize=20)
    plt.xticks([], fontsize=18)
    plt.yticks([0., 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.3)
    plt.xlim(0, np.max([t1[-1], t2[-1], t3[-1]]))
    plt.title('Mb: Rhythmic movement (novice)',
              fontsize=20)
    plt.text(0.5, 1.1, "(B)", fontsize=20)

    plt.subplot(313)
    plt.plot(t3, m3, '0.0', lw=2, label="Position")
    # # Line for targets P and Q.
    # plt.plot([0, t_max], [-0.025, -0.025], '0.0', linestyle='--', lw=0.5)
    # plt.plot([0, t_max], [0.025, 0.025], '0.0', linestyle='--', lw=0.5)
    # plt.plot([0, t_max], [1.025, 1.025], '0.0', linestyle='--', lw=0.5)
    # plt.plot([0, t_max], [0.975, 0.975], '0.0', linestyle='--', lw=0.5)
    plt.xlabel('Time (sec)', fontsize=20)
    plt.ylabel('Position', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks([0., 0.5, 1.0], fontsize=18)
    plt.ylim(-0.1, 1.3)
    plt.xlim(0, t_max)
    plt.title('Mc: Rhythmic movement (expert 2)',
              fontsize=20)
    plt.text(0.5, 1.1, "(C)", fontsize=20)

    plt.tight_layout()
    return fig
