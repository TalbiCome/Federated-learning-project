import numpy as np
import matplotlib.pyplot as plt

# Example data for fedAvg
accuracy_fedAvg = [0.1, 0.1, 0.1014, 0.0998, 0.1002, 0.1, 0.0996, 0.116, 0.1, 0.1, 0.1, 0.1016, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1192, 0.1022, 0.1, 0.1, 0.1, 0.1, 0.1184, 0.1, 0.1, 0.1314, 0.1274, 0.1, 0.1, 0.1326, 0.1454, 0.1496, 0.101, 0.1194, 0.152, 0.106, 0.1576, 0.1492, 0.155, 0.114, 0.1306, 0.1052, 0.1188, 0.1132]
numberOfRounds = [i for i in range(1, 51)]

# Example data for fedAdam
accuracy_fedAdam = [0.127, 0.1012, 0.1, 0.112, 0.1366, 0.1208, 0.1132, 0.1272, 0.1, 0.1, 0.1414, 0.139, 0.1088, 0.1, 0.1002, 0.1372, 0.1814, 0.137, 0.1486, 0.131, 0.1002, 0.1312, 0.153, 0.1288, 0.1814, 0.0934, 0.0934, 0.1006, 0.1386, 0.158, 0.1294, 0.142, 0.1452, 0.1416, 0.2, 0.186, 0.1668, 0.1498, 0.1626, 0.1488, 0.163, 0.1244, 0.2094, 0.1606, 0.1544, 0.1332, 0.1948, 0.1262, 0.1452, 0.1324]

# Plotting the graph
plt.plot(numberOfRounds, accuracy_fedAvg, marker='o', label='fedAvg')
plt.plot(numberOfRounds, accuracy_fedAdam, marker='s', label='fedAdam')
plt.xlabel('Number of Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Rounds')
plt.grid(True)
plt.legend()

# Save the graph as a PNG image
plt.savefig('/workspaces/ubuntuDevbox/workspace/Project/accuracy_vs_rounds.png')