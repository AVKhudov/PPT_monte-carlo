import numpy as np
import os
import matplotlib.pyplot as plt

# Файл работает с траекториями (вывод массивов и графиков)

path = r"D:\MC_Ion\18-08-2026_03-27-08\trajectories_18-08-2026_03-27-08\sort_18_gamma_threshold_200.0"

times = np.load(os.path.join(path, 'trajectory_t.npy'))
x_coordinates = np.load(os.path.join(path, 'trajectory_x.npy'))
p_xs = np.load(os.path.join(path, 'trajectory_p_x.npy'))
p_ys = np.load(os.path.join(path, 'trajectory_p_y.npy'))

number = 100

print(times[:, 0])

time = times[number]
x = x_coordinates[number]
p_x = p_xs[number]
p_y = p_ys[number]

#print(time)
#print(x)
#print(p_x)

plt.plot(time, p_x, label='p_x')
plt.plot(time, p_y, label='p_y')
plt.legend()
plt.show()
