import numpy as np

from scripts.trace_session_failures import channelwise_shift_fit


def test_channelwise_shift_fit_recovers_contact_specific_lags():
    n_samples = 160
    center = 80
    offsets = (1, -1)
    x = np.arange(61, dtype=np.float32) - 30
    template = np.stack((np.exp(-(x / 4) ** 2),
                         .8 * np.exp(-((x - 2) / 5) ** 2)), axis=1)
    residual = np.zeros((n_samples, 2), np.float32)
    for channel, lag in enumerate(offsets):
        residual[center + lag - 30:center + lag + 31, channel] = template[:, channel]

    score, amplitude, gain, fitted_offsets = channelwise_shift_fit(
        residual, center, template, np.array([0, 1]), np.ones(2))

    assert score > .99999
    assert amplitude == 1.
    assert gain > 0.
    assert fitted_offsets == list(offsets)

