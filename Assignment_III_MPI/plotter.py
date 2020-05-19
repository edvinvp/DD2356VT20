import matplotlib.pyplot as plt
import scipy.stats as stats

## Data /s, min from 3 runs.
processes = [8,16,32,64,128]
ex_2_1 = [3.860534, 1.967612, 1.075120, 0.613516, 0.314541]
ex_2_2 = [3.857900, 1.975822, 1.069269, 0.633160, 0.310544]
ex_2_3 = [3.857682, 1.974275, 1.071969, 0.614091, 0.313789]
ex_2_4 = [3.861319, 1.963683, 1.070561, 0.617922, 0.310366]
ex_2_5 = [3.861312, 1.964912, 1.068210, 0.641480, 0.310923]
ex_2_6 = [3.686523, 1.965352, 1.071266, 0.619348, 0.322147]

plt.plot(processes, ex_2_1, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_2, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_3, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_4, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_5, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_6, linestyle='--', marker='o')
plt.xlabel("Processes")
plt.ylabel("Time sec")
plt.show()
plt.plot(processes, ex_2_1, linestyle='--', marker='o', label="2.1")
plt.plot(processes, ex_2_2, linestyle='--', marker='o', label="2.2")
plt.plot(processes, ex_2_3, linestyle='--', marker='o', label="2.3")
plt.plot(processes, ex_2_4, linestyle='--', marker='o', label="2.4")
plt.plot(processes, ex_2_5, linestyle='--', marker='o', label="2.5")
plt.plot(processes, ex_2_6, linestyle='--', marker='o', label="2.6")
plt.legend()
plt.show()

ex_3_size = [2**x for x in range(3, 3+28)]

ex_3_hyperthreading_time = [0.000000610, 0.000000653, 0.000000684, 0.000000861, 0.000000830, 0.000000961, 0.000000856, 0.000000985, 0.000001154, 0.000001900, 0.000003958, 0.000004199, 0.000006795, 0.000011678, 0.000024951, 0.000045841, 0.000087616, 0.000172591, 0.000342784, 0.000686100, 0.001372836, 0.002740927, 0.005637565, 0.011530125, 0.023284757, 0.046784692, 0.093665702, 0.187317367]

ex_3_two_nodes_time = [0.000001342, 0.000001278, 0.000001278, 0.000001287, 0.000001299, 0.000001333, 0.000001431, 0.000001671, 0.000002062, 0.000002799, 0.000004907, 0.000005565, 0.000007534, 0.000012019, 0.000019531, 0.000034759, 0.000065529, 0.000126870, 0.000253870, 0.000465968, 0.000997248, 0.002059309, 0.004188287, 0.008719380, 0.017676899, 0.035695069, 0.071786742, 0.143827133]

# Do linear regression on the ex 3 data, inverse of slope = bandwidth,
# intercept = latency.
hyper_slope, hyper_intercept, _, _, _ = stats.linregress(ex_3_size, ex_3_hyperthreading_time)

two_nodes_slope, two_nodes_intercept, _, _, _ = stats.linregress(ex_3_size, ex_3_two_nodes_time)

print("EX 3: HYPERTHREADING: BANDWIDTH = " + str((1 / hyper_slope) / 1000000000)+ " GB/s")
print("EX 3: HYPERTHREADING: LATENCY  = " + str(hyper_intercept))
print("EX 3: TWO NODES: BANDWIDTH = " + str((1 / two_nodes_slope)/1000000000) + " GB/s")
print("EX 3: TWO NODES: LATENCY  = " + str(two_nodes_intercept))
plt.plot(ex_3_size, ex_3_two_nodes_time)
plt.xlabel("Message size")
plt.ylabel("Time sec")
plt.show()
plt.plot(ex_3_size, ex_3_hyperthreading_time)
plt.xlabel("Message size")
plt.ylabel("Time sec")
plt.show()
