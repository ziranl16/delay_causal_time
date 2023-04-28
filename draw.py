import matplotlib.pyplot as plt

data = [0.0677, 0.0786, 0.0785, 0.0806, 0.0997, 0.1124, 0.1185, 0.0889, 0.0884,
         0.0980, 0.0886]

# Create a bar plot
plt.bar(range(len(data)), data)

# Set the labels for the x-axis
plt.xticks(range(len(data)))

# Set the title and labels for the plot
plt.title("Plot of Given Data")
plt.xlabel("Index")
plt.ylabel("Value")

# Display the plot
plt.show()
plt.savefig('output_DEM.pdf')
