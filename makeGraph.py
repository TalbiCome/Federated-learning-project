import numpy as np
import matplotlib.pyplot as plt

# Example data for fedAvg
accuracy_fedAvg = [0.8, 0.85, 0.9, 0.95, 0.97]
numberOfRounds = [1, 2, 3, 4, 5]

# Example data for fedAdam
accuracy_fedAdam = [0.75, 0.82, 0.88, 0.93, 0.96]

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