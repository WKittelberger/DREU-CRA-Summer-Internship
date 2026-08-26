import numpy as np

from N_pulse_sequence import (
    build_n_pulse_sequence,
    NoisyFreeEvolutionPulseSequence,
    compute_toggling_frame_phase,
)


class ConstantNoiseModel:
    """
    Stand-in for qitcat's FrequencyNoise: returns a constant (DC)
    frequency offset instead of a stochastic realization. Used to test
    against the textbook spin-echo result: a symmetric spin echo should
    fully cancel a static detuning, while plain Ramsey should not.
    """

    def __init__(self, delta_nu_hz):
        self.delta_nu_hz = delta_nu_hz

    def generate_timeseries(self, duration, dt, n_realizations=1):
        n = int(np.ceil(duration / dt))
        t = np.arange(n) * dt
        noise = np.full((n_realizations, n), self.delta_nu_hz)
        return t, noise


def run_with_constant_offset(n_pi_pulses, delta_nu_hz, pulse_duration=1e-9, T_total=20e-3):
    pulse_areas, free_times, phase_offsets = build_n_pulse_sequence(
        n_pi_pulses=n_pi_pulses,
        pulse_duration=pulse_duration,
        T_total=T_total,
        readout_phase=np.pi / 2,
    )
    noise = ConstantNoiseModel(delta_nu_hz)
    seq = NoisyFreeEvolutionPulseSequence(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_times,
        noise_model=noise,
        noise_dt=1e-7,
        pulse_dt=1e-10,
    )
    _, prob = seq.run(pulse_phase_offsets=phase_offsets)
    return prob


def test_spin_echo_cancels_static_offset():
    """N=1 (single mirror pulse, classic spin echo) with a symmetric
    T-pi-T spacing should fully refocus a static/DC detuning: P_e stays
    at the mid-fringe value ~0.5 regardless of the offset size."""
    for delta_nu in [50.0, 500.0, 5000.0]:
        prob = run_with_constant_offset(n_pi_pulses=1, delta_nu_hz=delta_nu)
        assert np.isclose(prob, 0.5, atol=1e-3), (
            f"spin echo failed to cancel static offset {delta_nu} Hz: P_e={prob}"
        )
    print("PASS: spin echo (N=1) cancels static offset at all tested magnitudes")


def test_ramsey_does_not_cancel_static_offset():
    """N=0 (plain Ramsey, no mirror pulse) should NOT cancel a static
    offset -- P_e should move well away from 0.5."""
    prob = run_with_constant_offset(n_pi_pulses=0, delta_nu_hz=37.0)
    assert not np.isclose(prob, 0.5, atol=1e-3), (
        f"Ramsey unexpectedly canceled a static offset: P_e={prob}"
    )
    print(f"PASS: Ramsey (N=0) does not cancel static offset (P_e={prob:.4f})")


def test_matches_toggling_frame_reference():
    """In the instantaneous-pulse limit, the *noise-induced* phase from
    the full finite-duration simulation should match the idealized
    toggling-frame integral, for a real stochastic noise realization.

    Note: raw state phase also contains a fixed geometric phase from
    the pi-pulses themselves (each ideal pi-pulse imprints a -i
    prefactor, independent of noise), so we isolate the noise-induced
    part by differencing against a zero-noise run of the identical
    sequence rather than comparing raw phases directly.
    """
    from qitcat.modules.noise import FrequencyNoise

    n_pi = 3
    pulse_duration = 1e-9  # ~instantaneous
    T_total = 15e-3

    pulse_areas, free_times, phase_offsets = build_n_pulse_sequence(
        n_pi_pulses=n_pi,
        pulse_duration=pulse_duration,
        T_total=T_total,
        readout_phase=np.pi / 2,
    )

    # --- noisy run ---
    noise = FrequencyNoise(ILW=864, f_k=1.16e6, noise_fraction=0.05, seed=99)
    seq_noisy = NoisyFreeEvolutionPulseSequence(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_times,
        noise_model=noise,
        noise_dt=1e-7,
        pulse_dt=1e-10,
    )
    state_noisy, _ = seq_noisy.run(pulse_phase_offsets=phase_offsets)
    phase_noisy = np.angle(state_noisy[1]) - np.angle(state_noisy[0])

    # --- zero-noise run of the identical sequence (isolates geometric phase) ---
    zero_noise = ConstantNoiseModel(0.0)
    seq_zero = NoisyFreeEvolutionPulseSequence(
        pulse_areas=pulse_areas,
        pulse_duration=pulse_duration,
        free_evolution_time=free_times,
        noise_model=zero_noise,
        noise_dt=1e-7,
        pulse_dt=1e-10,
    )
    state_zero, _ = seq_zero.run(pulse_phase_offsets=phase_offsets)
    phase_zero = np.angle(state_zero[1]) - np.angle(state_zero[0])

    # Noise-induced phase only, wrapped to (-pi, pi]
    simulated_phase = np.angle(np.exp(1j * (phase_noisy - phase_zero)))

    reference_phase = compute_toggling_frame_phase(
        seq_noisy.noise_t, seq_noisy.noise_hz, pulse_areas, pulse_duration, free_times
    )
    reference_phase_wrapped = np.angle(np.exp(1j * reference_phase))

    diff = abs(
        np.angle(np.exp(1j * (simulated_phase - reference_phase_wrapped)))
    )
    print(
        f"simulated_phase={simulated_phase:.6f} rad, "
        f"reference_phase={reference_phase_wrapped:.6f} rad, diff={diff:.2e}"
    )
    assert diff < 0.05, "finite-duration simulation deviates from toggling-frame reference"
    print("PASS: finite-duration simulation matches toggling-frame reference "
          "in the instantaneous-pulse limit")


if __name__ == "__main__":
    test_spin_echo_cancels_static_offset()
    test_ramsey_does_not_cancel_static_offset()
    test_matches_toggling_frame_reference()
