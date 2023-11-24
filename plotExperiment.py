import matplotlib.pyplot as plt

#PART1 PLOTS
img_size = [128,320,416,496]
accuracy = [0.024211, 0.014176, 0.013530, 0.013286]


# Plotting
plt.plot(img_size, accuracy, marker='o')
plt.title('Image size vs Performance')
plt.xlabel('Image size')
plt.ylabel('Accuracy')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/Part1/size_vs_performance_plot.png')

# Display the plot (optional)
plt.show()


plt.clf()



#PART 2 PLOTS
num_samples = [85, 170, 255, 340, 425]
accuracy = [0.166641, 0.098490, 0.056021, 0.040865, 0.031671]


# Plotting
plt.plot(num_samples, accuracy, marker='o')
plt.title('Number of Samples vs Performance')
plt.xlabel('Number of Samples')
plt.ylabel('Accuracy')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/Part2/samples_vs_performance_plot.png')

# Display the plot (optional)
plt.show()