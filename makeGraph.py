import numpy as np
import matplotlib.pyplot as plt

# Example data for fedAvg
accuracy_fedAvg = [0.1, 0.1, 0.1014, 0.0998, 0.1002, 0.1, 0.0996, 0.116, 0.1, 0.1, 0.1, 0.1016, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1192, 0.1022, 0.1, 0.1, 0.1, 0.1, 0.1184, 0.1, 0.1, 0.1314, 0.1274, 0.1, 0.1, 0.1326, 0.1454, 0.1496, 0.101, 0.1194, 0.152, 0.106, 0.1576, 0.1492, 0.155, 0.114, 0.1306, 0.1052, 0.1188, 0.1132]
numberOfRounds = [i for i in range(1, 51)]

# Example data for fedAdam
accuracy_fedAdam = [0.1, 0.0876, 0.1, 0.1022, 0.1, 0.1, 0.1322, 0.1, 0.1264, 0.1006, 0.1, 0.1, 0.1004, 0.1412, 0.1, 0.1, 0.1474, 0.1066, 0.1124, 0.1048, 0.0966, 0.1, 0.1016, 0.1, 0.1196, 0.1, 0.1284, 0.1274, 0.1142, 0.103, 0.1148, 0.1112, 0.128, 0.177, 0.15, 0.1268, 0.1354, 0.1098, 0.187, 0.1664, 0.1816, 0.1906, 0.1666, 0.152, 0.1954, 0.1688, 0.1448, 0.1562, 0.166, 0.1572]

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