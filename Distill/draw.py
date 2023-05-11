import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

total_epochs = 60
sampling_points = [i * 10 for i in range(total_epochs)]
gamma_values = []
num_training_batches = 10
# plt.rcParams['font.sans-serif']=['Times New Roman']

for epoch in range(total_epochs):
    for i in range(num_training_batches):
        current_step = epoch * num_training_batches + i
        total_steps = total_epochs * num_training_batches
        multiplier = min(1, 0.75 * (1 - math.cos(math.pi * current_step / total_steps)))
        gamma_values.append(multiplier)


# drwa
arr = np.array(range(total_steps))
float_arr = arr.astype(np.float64) / 10
plt.plot(float_arr, gamma_values,'black',linewidth=2.5)
plt.xlabel('Epoch')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10)) 
plt.ylabel('Gamma')
plt.title('Gamma values during training')

plt.savefig("./figs/gama")
