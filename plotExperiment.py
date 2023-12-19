import matplotlib.pyplot as plt

#PART1 PLOTS (Size vs accuracy)
img_size = [128,320,416,496]
accuracy = [0.948294, 0.976131, 0.975655,0.981708]


# Plotting
plt.plot(img_size, accuracy, marker='o')
plt.title('Image size vs Performance')
plt.xlabel('Image size')
plt.ylabel('Validation Loss')
plt.grid(True)

# Save the plot as a PNG file
#plt.savefig('./results/Part1/size_vs_accuracy_plot.png')

plt.show()


plt.clf()


#PART 2 PLOTS (Num samples vs accuracy) 
num_samples = [85, 170, 255, 340, 425]
accuracy = [0.940763, 0.965152, 0.971517, 0.968903, 0.972835]


# Plotting
plt.plot(num_samples, accuracy, marker='o')
plt.title('Number of Samples vs accuracy')
plt.xlabel('Number of Samples')
plt.ylabel('Pixel Accuracy')
plt.grid(True)

# Save the plot as a PNG file
#plt.savefig('./results/Part2/samples_vs_accuracy_plot.png')

# Display the plot (optional)
plt.show()


plt.clf()

#Part 3 PLOT (Num pixels vs accuracy)
# Data for the first set (Image Size)
pixels_blue = [32*32*425,64*64*425,128 * 128 * 425, 256*256*425,320 * 320 * 425, 352*352*425,416 * 416 * 425, 496 * 496 * 425]
accuracy_blue = [0.904102,0.948137, 0.948294, 0.963689, 0.976131,0.976054, 0.975655, 0.981708]

# Data for the second set (Number of training samples)
pixels_red = [85 * 320 * 320, 170 * 320 * 320, 255 * 320 * 320, 340 * 320 * 320, 425 * 320 * 320,340*496*496, 255*496*496]
accuracy_red = [0.940763, 0.965152, 0.971517, 0.968903, 0.972835,0.976764,0.978264]

# Plotting
plt.scatter(pixels_blue, accuracy_blue, color='blue', label='Size')
plt.scatter(pixels_red, accuracy_red, color='red', label='Samples')

# Adding labels and title
plt.xlabel('Pixels')
plt.ylabel('Accuracy')
plt.title('Pixel vs Accuracy')

# Adding a legend
plt.legend()

plt.savefig('./results/pixels_vs_accuracy_scatterplot.png')

# Display the plot
plt.show()