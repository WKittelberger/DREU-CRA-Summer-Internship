import numpy as np

from qitcat.modules.noise import FrequencyNoise

from pulse_sequence_frequency import (
    FrequencyNoisePulseSequence
)

# Laser frequency-noise model

noise = FrequencyNoise(
    ILW=864,
    f_k=1.16e6,
    noise_fraction=0.05,
)

# Mach-Zehnder sequence

sequence = FrequencyNoisePulseSequence(

    pulse_areas=[
        np.pi / 2,
        np.pi,
        np.pi / 2,
    ],

    pulse_duration=10e-6,

    free_evolution_time=10e-3,

    noise_model=noise,

    noise_dt=1e-7,

    pulse_dt=1e-7,
)

#run

state, probability = sequence.run()


print("Final state:")
print(state)

print()
print("Final excited-state probability:")
print(probability)