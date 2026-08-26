"""
Quadrature-corrected transfer function (filter function) measurement,
plus noise overlap integral that turns filter function into a predicted 
phase noise variance for a given noise PSD.

Why we need quadrature correction
------------------------------------
filter_function.py's compute_filter_function measures the response to
a COSINE-phase tone only, i.e. it recovers Re{G(f)}, where G(f) is the
sequence's complex frequency-domain sensitivity (the Fourier transform
of its real-time noise-sensitivity kernel g(t), which reduces to
2*pi*y(t) in the idealized instantaneous-pulse limit). Because g(t) is
not generally symmetric about the sequence's midpoint, G(f) can have a
meaningful imaginary (sine-phase) part too -- using only the cosine
measurement can under-estimate the true |G(f)|, which is what a
noise-overlap integral actually needs:

    sigma_phi^2 = integral_0^inf S(f) * |G(f)|^2 df           (Parseval)

This module measures BOTH quadratures (injecting a cosine tone and a
sine tone at each frequency) and combines them:

    |G(f)|^2 = Re{G(f)}^2 + Im{G(f)}^2
--------------------------------
    - N-pulse study: sequence_cls=N_pulse_sequence.NoisyFreeEvolutionPulseSequence
      (noise couples during both free evolution AND pulses. Full CPMG-decoupling model)
    - pulse duration study: sequence_cls=N_pulse_sequence.FastFrequencyNoisePulseSequence
      (noise couples ONLY during the finite pulses -- isolates the
      in-pulse-only mechanism, cleanly separated from CPMG decoupling)

Validated against real Monte Carlo in test_filter_quadrature.py.
"""

import numpy as np

from N_pulse_sequence import build_n_pulse_sequence, _trapz
from filter_function import ToneNoiseModel, ZeroNoiseModel, _extract_phi_from_probability


def measure_response_quadrature(
    sequence_cls,
    n_pi_pulses,
    pulse_duration,
    T_total,
    freq_hz,
    amplitude_hz=1.0,
    noise_dt=None,
    pulse_dt=1e-8,
    mode="fixed_total_time",
    segment_spacing=None,
):
    """
    Measure |G(f)| at a single frequency using two tone injections, 
    combined in quadrature. Each is a deterministic simulation.

    Returns
    -------
    magnitude :
        |G(f)| / amplitude_hz (rad/Hz), the quadrature-combined
        transfer function magnitude at this frequency.
    """
    pulse_areas, free_evolution_time, phase_offsets = build_n_pulse_sequence(
        n_pi_pulses=n_pi_pulses,
        pulse_duration=pulse_duration,
        T_total=T_total,
        segment_spacing=segment_spacing,
        mode=mode,
        readout_phase=np.pi / 2,
    )

    if noise_dt is None:
        noise_dt = min(1e-7, 1.0 / (40.0 * max(freq_hz, 1.0)))

    # Zero-noise baseline (shared across both quadratures; removes any
    # residual systematic bias from finite pulse-area/duration effects)
    zero = ZeroNoiseModel()
    seq_zero = sequence_cls(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_evolution_time,
        noise_model=zero,
        noise_dt=noise_dt,
        pulse_dt=pulse_dt,
    )
    _, prob_zero = seq_zero.run(pulse_phase_offsets=phase_offsets)
    phi_zero = _extract_phi_from_probability(prob_zero)

    quadrature_components = []
    for phase0 in (0.0, -np.pi / 2): 
        tone = ToneNoiseModel(amplitude_hz=amplitude_hz, freq_hz=freq_hz, phase0=phase0)
        seq_tone = sequence_cls(
            pulse_areas=pulse_areas,
            pulse_duration=pulse_duration,
            free_evolution_time=free_evolution_time,
            noise_model=tone,
            noise_dt=noise_dt,
            pulse_dt=pulse_dt,
        )
        _, prob_tone = seq_tone.run(pulse_phase_offsets=phase_offsets)
        phi_tone = _extract_phi_from_probability(prob_tone)
        quadrature_components.append((phi_tone - phi_zero) / amplitude_hz)

    g_cos, g_sin = quadrature_components
    return float(np.hypot(g_cos, g_sin))


def compute_filter_function_quadrature(
    sequence_cls,
    n_pi_pulses,
    pulse_duration,
    T_total,
    freqs_hz,
    amplitude_hz=1.0,
    **kwargs,
):
    """Sweep measure_response_quadrature over an array of frequencies."""
    response = np.empty(len(freqs_hz))
    for i, f in enumerate(freqs_hz):
        response[i] = measure_response_quadrature(
            sequence_cls, n_pi_pulses, pulse_duration, T_total, f, amplitude_hz, **kwargs
        )
    return response


def lorentzian_psd(freqs_hz, ILW, f_k):
    """
    qitcat FrequencyNoise's NOMINAL PSD shape: S(f) = ILW / sqrt(1 + (f/f_k)^2).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    return ILW / np.sqrt(1.0 + (freqs_hz / f_k) ** 2)


def measure_empirical_psd(ILW, f_k, noise_fraction, duration=2.0, dt=2e-5,
                           n_realizations=40, nperseg=8192):
    """
    NOTE on defaults: `duration` and `dt` here are chosen for accurate
    PSD ESTIMATION (need long duration and modest sample rate for good
    low-frequency resolution: df = 1/(dt*N_segment) ~ 1/(dt*nperseg)),
    which is a different requirement than the fine dt needed for the
    quantum pulse simulation elsewhere in this project. Do not reuse the
    simulation's noise_dt (e.g. 1e-7) here -- with nperseg=4096 that
    gives ~2.4 kHz frequency resolution, which entirely misses any
    f_k below a few kHz. Defaults here give ~6 Hz resolution up to
    25 kHz Nyquist, adequate for the f_k ranges used in both studies in
    this project; widen/narrow duration and dt if you need to resolve a
    different range.

    Returns
    -------
    freqs_hz :
    mean_psd :
        Welch-estimated PSD [Hz^2/Hz], averaged over n_realizations
        independent draws.
    """
    from scipy.signal import welch
    from qitcat.modules.noise import FrequencyNoise

    psds = []
    freqs_hz = None
    for _ in range(n_realizations):
        fn = FrequencyNoise(ILW=ILW, f_k=f_k, noise_fraction=noise_fraction)
        t, ts = fn.generate_timeseries(duration=duration, dt=dt, n_realizations=1)
        f, psd = welch(ts, fs=1.0 / dt, nperseg=nperseg)
        freqs_hz = f
        psds.append(psd)
    return freqs_hz, np.mean(psds, axis=0)


def predict_sigma_phi_squared(freqs_hz, response_quadrature, psd):
    """
    filter-function noise-overlap integral:

        sigma_phi^2 ~= integral_0^inf S(f) * |G(f)|^2 df

    IMPORTANT: freqs_hz must span from near-DC up through well past
    where response_quadrature has decayed to negligible levels relative
    to the peak of S(f)*response^2, or the integral will be truncated
    and sigma_phi underestimated. Validated against real Monte Carlo in
    test_filter_quadrature.py.
    """
    integrand = np.asarray(psd) * np.asarray(response_quadrature) ** 2
    return float(_trapz(integrand, freqs_hz))


def resample_response_to_psd_grid(freqs_hz_response, response_quadrature, freqs_hz_psd):
    """
    Linearly interpolate a response curve onto the linear grid Welch's method
    returns for the PSD, so predict_sigma_phi_squared can be evaluated
    on a single common grid. Extrapolates with edge values.
    """
    return np.interp(freqs_hz_psd, freqs_hz_response, response_quadrature)
