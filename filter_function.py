"""
Filter-function (frequency response) calculator for CPMG (N pulse) sequences.

if the laser frequency were wobbling at a single frequency f, how much phase 
would this sequence pick up?"

That's done by literally feeding a monochromatic tone through the
sequence simulator (NoisyFreeEvolutionPulseSequence),
including finite pulse duration effects, and reading out the resulting
phase. Sweeping f traces out the sequence's filter function 
which frequencies it passes and which it rejects. This determines 
whether/how CPMG suppression shows up for a given noise spectrum.

Each point on the curve is a single deterministic simulation."""

import numpy as np

from N_pulse_sequence import (
    build_n_pulse_sequence,
    NoisyFreeEvolutionPulseSequence,
)


class ToneNoiseModel:

    def __init__(self, amplitude_hz, freq_hz, phase0=0.0):
        self.amplitude_hz = amplitude_hz
        self.freq_hz = freq_hz
        self.phase0 = phase0

    def generate_timeseries(self, duration, dt, n_realizations=1):
        n = int(np.ceil(duration / dt))
        t = np.arange(n) * dt
        tone = self.amplitude_hz * np.cos(2.0 * np.pi * self.freq_hz * t + self.phase0)
        noise = np.tile(tone, (n_realizations, 1))
        return t, noise


class ZeroNoiseModel:
    """subtract off the fixed geometric phase that ideal pulses imprint, isolates noise 
    induced phase response."""

    def generate_timeseries(self, duration, dt, n_realizations=1):
        n = int(np.ceil(duration / dt))
        t = np.arange(n) * dt
        noise = np.zeros((n_realizations, n))
        return t, noise


def _extract_phi_from_probability(probability):
    """
    Exact inversion of the mid-fringe readout relation
    P_e = 0.5 + 0.5*sin(phi)  =>  phi = arcsin(2*(P_e - 0.5)).

    This is not a small-angle approximation for |phi| <= pi/2.
    Only valid up to the arcsin branch ambiguity at |phi|=pi/2,
    so amplitude_hz in measure_phase_response should be kept small
    enough that the response stays within that range."""

    return np.arcsin(np.clip(2.0 * (probability - 0.5), -1.0, 1.0))


def measure_phase_response(
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
    Inject a single tone at freq_hz and return the resulting
    noise-induced phase (rad), extracted via the exact mid-fringe
    inversion phi = arcsin(2*(P_e - 0.5)) (valid for |phi| <= pi/2.
    Zero-noise run subtracted to remove any residual bias.

    Parameters:
    freq_hz :
        Tone frequency to test (Hz).
    amplitude_hz :
        Tone amplitude (Hz).
    noise_dt : 
        Sampling step for the tone timeseries.
    pulse_dt : 
        Integration step during pulses.

    Returns
    -------
    phi : float
        Noise-induced accumulated phase (rad) at this tone frequency.
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

    tone = ToneNoiseModel(amplitude_hz=amplitude_hz, freq_hz=freq_hz)
    seq_tone = NoisyFreeEvolutionPulseSequence(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_evolution_time,
        noise_model=tone,
        noise_dt=noise_dt,
        pulse_dt=pulse_dt,
    )
    _, prob_tone = seq_tone.run(pulse_phase_offsets=phase_offsets)
    phi_tone = _extract_phi_from_probability(prob_tone)

    # Zero-noise baseline: ideally P_e is exactly 0.5 at mid-fringe with
    # no noise, but subtract it anyway to absorb any residual systematic
    # bias from finite pulse-area/duration effects, keeping the result
    # purely noise-induced.
    zero = ZeroNoiseModel()
    seq_zero = NoisyFreeEvolutionPulseSequence(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_evolution_time,
        noise_model=zero,
        noise_dt=noise_dt,
        pulse_dt=pulse_dt,
    )
    _, prob_zero = seq_zero.run(pulse_phase_offsets=phase_offsets)
    phi_zero = _extract_phi_from_probability(prob_zero)

    return phi_tone - phi_zero


def compute_filter_function(
    n_pi_pulses,
    pulse_duration,
    T_total,
    freqs_hz,
    amplitude_hz=1.0,
    mode="fixed_total_time",
    segment_spacing=None,
):
    """
    Sweep measure_phase_response over an array of frequencies.

    Returns
    response : 
        The transfer function magnitude, in rad per Hz of tone amplitude.
        (only magnitudenmatters for characterizing which frequencies pass/rejected.)
    """
    response = np.empty(len(freqs_hz))
    for i, f in enumerate(freqs_hz):
        phi = measure_phase_response(
            n_pi_pulses=n_pi_pulses,
            pulse_duration=pulse_duration,
            T_total=T_total,
            freq_hz=f,
            amplitude_hz=amplitude_hz,
            mode=mode,
            segment_spacing=segment_spacing,
        )
        response[i] = abs(phi) / amplitude_hz
    return response
