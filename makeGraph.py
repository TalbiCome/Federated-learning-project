import numpy as np
import matplotlib.pyplot as plt

# Example data for fedAvg
accuracy_fedAvg = [0.1, 0.1002, 0.1016, 0.1046, 0.124, 0.149, 0.1176, 0.1748, 0.1162, 0.1722, 0.101, 0.2004, 0.129, 0.1516, 0.1236, 0.1824, 0.1922, 0.1642, 0.13, 0.1976, 0.1684, 0.1852, 0.1198, 0.1836, 0.195, 0.144, 0.1908, 0.2096, 0.2242, 0.233, 0.2504, 0.2404, 0.2582, 0.2, 0.2328, 0.2194, 0.2424, 0.2376, 0.2394, 0.1804, 0.2096, 0.1646, 0.1944, 0.2298, 0.2386, 0.2122, 0.2648, 0.229, 0.2002, 0.1916]
numberOfRounds = [i for i in range(1, 51)]

# Example data for fedAdam
accuracy_fedAdam = [0.1, 0.1, 0.1, 0.1202, 0.1238, 0.1, 0.1336, 0.1486, 0.1518, 0.1474, 0.149, 0.1572, 0.145, 0.12, 0.138, 0.1772, 0.1962, 0.2136, 0.21, 0.2154, 0.2272, 0.2396, 0.2494, 0.2702, 0.245, 0.244, 0.2628, 0.2716, 0.2664, 0.2778, 0.2822, 0.29, 0.2934, 0.3008, 0.3046, 0.3096, 0.3004, 0.2906, 0.2954, 0.2924, 0.291, 0.297, 0.3004, 0.3076, 0.3144, 0.3026, 0.297, 0.306, 0.3238, 0.3376]

# Plotting the graph
plt.plot(numberOfRounds, accuracy_fedAvg, marker='o', label='fedAvg')
plt.plot(numberOfRounds, accuracy_fedAdam, marker='s', label='fedAdam')
plt.xlabel('Number of Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy on non iid class dataset')
plt.grid(True)
plt.legend()

# Save the graph as a PNG image
plt.savefig('/workspaces/ubuntuDevbox/workspace/Project/accuracy_vs_rounds.png')