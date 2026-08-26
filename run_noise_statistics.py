import numpy as np
import matplotlib.pyplot as plt

from qitcat.modules.noise import FrequencyNoise

from pulse_sequence_frequency import (
    FrequencyNoisePulseSequence
)

# Simulation parameters

N_RUNS = 1000

PULSE_DURATION = 10e-6       # 10 microseconds
FREE_EVOLUTION_TIME = 10e-3  # 10 ms

# Mach-Zehnder sequence:
#
#       pi/2 --- pi --- pi/2

PULSE_AREAS = [
    np.pi / 2,
    np.pi,
    np.pi / 2,
]

probabilities = []
final_states = []

# Run many noise realizations

for run_number in range(N_RUNS):

    noise = FrequencyNoise(
        ILW=864,
        f_k=1.16e6,
        noise_fraction=0.05,
    )

    # Create a new pulse-sequence experiment
    sequence = FrequencyNoisePulseSequence(

        pulse_areas=PULSE_AREAS,

        pulse_duration=PULSE_DURATION,

        free_evolution_time=FREE_EVOLUTION_TIME,

        noise_model=noise,

        noise_dt=1e-7,

        pulse_dt=1e-7,
    )

    # Run the experiment
    state, probability = sequence.run()

    # Store results
    final_states.append(state)
    probabilities.append(probability)

probabilities = np.asarray(probabilities)

# Statistics

mean_probability = np.mean(probabilities)

variance_probability = np.var(
    probabilities,
    ddof=1
)

standard_deviation_probability = np.std(
    probabilities,
    ddof=1
)


print()
print("=" * 40)
print("LASER FREQUENCY NOISE STATISTICS")
print("=" * 40)

print(f"Number of realizations: {N_RUNS}")
print(f"Pulse duration:         {PULSE_DURATION:.2e} s")
print(f"Free evolution time:    {FREE_EVOLUTION_TIME:.2e} s")

print()
print(f"Mean P_e:               {mean_probability:.6e}")
print(f"Variance of P_e:        {variance_probability:.6e}")
print(f"Std. dev. of P_e:       {standard_deviation_probability:.6e}")

print()
print(f"Minimum P_e:            {np.min(probabilities):.6e}")
print(f"Maximum P_e:            {np.max(probabilities):.6e}")
print()

# Histogram

plt.figure(figsize=(8, 5))

plt.hist(
    probabilities,
    bins=40,
)

plt.xlabel("Final excited-state probability $P_e$")
plt.ylabel("Number of realizations")

plt.title(
    "Distribution of Final Excited-State Probability\n"
    "under Laser Frequency Noise"
)

plt.tight_layout()