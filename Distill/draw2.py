import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

total_epochs = 60
sampling_points = [i * 10 for i in range(total_epochs)]
acc = ([0.4512,0.4436,0.4737,0.4764,0.4912,0.4908,0.4992,0.4935,0.4872,0.4980,0.4970,0.5047])
bar = ([0.4581,0.4581,0.4581,0.4581,0.4581,0.4581,0.4581,0.4581,0.4581,0.4581,0.4581,0.4581])

# drwa
arr = np.array(range(0,total_epochs,5)) +5
float_arr = arr.astype(np.float64)
print(arr)
plt.plot(arr, acc,'red',linewidth=2,marker='s',label='Training Model')
plt.plot(arr, bar,'blue',linewidth=2,marker='o',linestyle='--',label='Pretrained Model')
plt.xlabel('Epoch')
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1)) 
plt.ylabel('acc')
plt.title('test accuracy on target domain')
plt.legend()
plt.savefig("./figs/testlog")