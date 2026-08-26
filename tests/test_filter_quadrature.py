import numpy as np

from filter_quadrature import measure_response_quadrature, compute_filter_function_quadrature
from filter_function import measure_phase_response
from N_pulse_sequence import NoisyFreeEvolutionPulseSequence


def test_quadrature_geq_cosine_only():
    """|G(f)| from the quadrature-combined measurement must be >= the
    cosine-only measurement's magnitude at every frequency (the cosine
    component is one leg of a right triangle whose hypotenuse is |G(f)|)."""
    for f in [10, 50, 137, 300]:
        cos_only = abs(measure_phase_response(2, 10e-6, 20e-3, freq_hz=f, amplitude_hz=1.0))
        quad = measure_response_quadrature(
            NoisyFreeEvolutionPulseSequence, 2, 10e-6, 20e-3, freq_hz=f, amplitude_hz=1.0
        )
        assert quad >= cos_only - 1e-12, f"f={f}: quadrature ({quad}) < cosine-only ({cos_only})"
    print("PASS: quadrature-combined response is always >= cosine-only response")


def test_quadrature_response_is_finite_and_positive():
    freqs = np.logspace(0, 4, 10)
    response = compute_filter_function_quadrature(
        NoisyFreeEvolutionPulseSequence, 1, 10e-6, 20e-3, freqs, amplitude_hz=1.0
    )
    assert np.all(np.isfinite(response))
    assert np.all(response >= 0)
    print("PASS: quadrature response is finite and non-negative across a frequency decade sweep")


if __name__ == "__main__":
    test_quadrature_geq_cosine_only()
    test_quadrature_response_is_finite_and_positive()
    print(
        "\nNOTE: absolute calibration of predict_sigma_phi_squared against\n"
        "qitcat's actual FrequencyNoise output was investigated but NOT\n"
        "fully resolved -- the discrepancy against real Monte Carlo changed\n"
        "with the (duration, dt) grid used to estimate the empirical PSD,\n"
        "suggesting FrequencyNoise.generate_timeseries's absolute scale is\n"
        "not simply grid-invariant. The methodology itself (quadrature\n"
        "correction, linearity, DC-rejection, toggling-frame agreement) is\n"
        "validated above and in test_filter_function.py. Treat\n"
        "predict_sigma_phi_squared's output as relative/qualitative\n"
        "guidance only until this is pinned down; use direct Monte Carlo\n"
        "(n_pulse_sequence.run_single_shot) for quantitative results, as both\n"
        "n_pulses_fk_study.py and pulse_duration_fk_study.py do."
    )
