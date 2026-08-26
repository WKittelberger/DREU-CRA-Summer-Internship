import numpy as np

from filter_function import measure_phase_response, compute_filter_function
from N_pulse_sequence import build_n_pulse_sequence, compute_toggling_frame_phase, _trapz
from filter_function import ToneNoiseModel


def test_linear_response():
    """Halving the tone amplitude should exactly halve the phase response
    (confirms we're in the linear regime for these parameters)."""
    phi_1 = measure_phase_response(2, 10e-6, 20e-3, freq_hz=50.0, amplitude_hz=0.5)
    phi_2 = measure_phase_response(2, 10e-6, 20e-3, freq_hz=50.0, amplitude_hz=0.25)
    assert np.isclose(phi_2, phi_1 / 2, rtol=1e-3), (phi_1, phi_2)
    print("PASS: response is linear in tone amplitude")


def test_ramsey_passes_dc_echo_rejects_it():
    """Near-DC (f << 1/T_total): Ramsey should respond ~maximally,
    a spin echo should reject it almost completely -- the same physics
    validated with a true static offset in test_free_evolution_noise.py,
    now recovered via the tone-injection method."""
    T_total = 20e-3
    f_dc = 1.0  # Hz, effectively static over a 20ms window
    expected = 2 * np.pi * 1.0 * T_total  # amplitude_hz=1.0

    phi_ramsey = measure_phase_response(0, 1e-9, T_total, freq_hz=f_dc, amplitude_hz=1.0)
    phi_echo = measure_phase_response(1, 1e-9, T_total, freq_hz=f_dc, amplitude_hz=1.0)

    assert np.isclose(phi_ramsey, expected, rtol=0.01), (phi_ramsey, expected)
    assert abs(phi_echo) < 0.01 * expected, (phi_echo, expected)
    print(f"PASS: Ramsey passes DC (phi={phi_ramsey:.5f}, expected {expected:.5f}); "
          f"echo rejects it (phi={phi_echo:.2e})")


def test_matches_toggling_frame_at_single_frequency():
    """In the instantaneous-pulse limit, the tone-injection result
    should match the idealized analytic toggling-frame calculation
    evaluated on the same tone realization."""
    n_pi = 3
    pulse_duration = 1e-9
    T_total = 15e-3
    freq_hz = 137.0
    amplitude_hz = 1.0

    phi_sim = measure_phase_response(
        n_pi, pulse_duration, T_total, freq_hz=freq_hz, amplitude_hz=amplitude_hz
    )

    pulse_areas, free_times, _ = build_n_pulse_sequence(
        n_pi, pulse_duration, T_total=T_total, readout_phase=np.pi / 2
    )
    tone = ToneNoiseModel(amplitude_hz, freq_hz)
    t, noise = tone.generate_timeseries(duration=T_total + 2 * pulse_duration, dt=1e-7)
    phi_ref = compute_toggling_frame_phase(t, noise[0], pulse_areas, pulse_duration, free_times)

    diff = abs(np.angle(np.exp(1j * (phi_sim - phi_ref))))
    print(f"phi_sim={phi_sim:.6f}  phi_ref={phi_ref:.6f}  diff={diff:.2e}")
    assert diff < 0.02, "tone-injection result deviates from toggling-frame reference"
    print("PASS: tone-injection matches toggling-frame reference in instantaneous-pulse limit")


def test_dc_response_zero_for_all_n_not_just_odd():
    """
    Regression test for the CPMG spacing bug
    """
    T_total = 20e-3
    f_dc = 1.0  # effectively static over a 20ms window
    expected_ramsey = 2 * np.pi * 1.0 * T_total  # N=0 scale, for a relative tolerance
    for n_pi in [1, 2, 3, 4, 8]:
        phi = measure_phase_response(n_pi, 1e-9, T_total, freq_hz=f_dc, amplitude_hz=1.0)
        assert abs(phi) < 0.01 * expected_ramsey, f"N={n_pi}: DC leak {phi:.2e} rad, expected ~0"
    print("PASS: DC response is ~zero for every N (not just odd N) -- CPMG spacing bug fixed")


if __name__ == "__main__":
    test_linear_response()
    test_ramsey_passes_dc_echo_rejects_it()
    test_matches_toggling_frame_at_single_frequency()
    test_dc_response_zero_for_all_n_not_just_odd()