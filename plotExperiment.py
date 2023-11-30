import matplotlib.pyplot as plt

#PART1 PLOTS (Size vs performance)
img_size = [128,320,416,496]
loss = [0.948294, 0.976131, 0.975655,0.981708]


# Plotting
plt.plot(img_size, loss, marker='o')
plt.title('Image size vs Performance')
plt.xlabel('Image size')
plt.ylabel('Validation Loss')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/Part1/size_vs_accuracy_plot.png')

# Display the plot (optional)
plt.show()


plt.clf()


#PART 2 PLOTS (Num samples vs performance) 
num_samples = [85, 170, 255, 340, 425]
accuracy = [0.940763, 0.965152, 0.971517, 0.968903, 0.972835]


# Plotting
plt.plot(num_samples, accuracy, marker='o')
plt.title('Number of Samples vs accuracy')
plt.xlabel('Number of Samples')
plt.ylabel('Pixel Accuracy')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/Part2/samples_vs_accuracy_plot.png')

# Display the plot (optional)
plt.show()


plt.clf()

#Part 3 PLOT (Num pixels vs performance)
num_pixels = [128*128*425,85*320*320, 170*320*320, 255*320*320, 340*320*320, 425*320*320,  320*320*425,416*416*425,496*496*425 ]
accuracy = [0.948294,0.940763, 0.965152, 0.971517, 0.968903, 0.972835, 0.976131, 0.975655,0.981708]

# Plotting
plt.plot(num_pixels, accuracy, marker='o')
plt.title('Number of Pixels vs accuracy')
plt.xlabel('Number of Pixels')
plt.ylabel('Pixel Accuracy')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/pixels_vs_accuracy_plot.png')

# Display the plot (optional)
plt.show()