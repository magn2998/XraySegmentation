import matplotlib.pyplot as plt

#PART1 PLOTS 1
img_size = [32, 64, 128, 256, 320, 352, 416, 496]
acc = [0.955880, 0.958846, 0.957834, 0.968581, 0.977910, 0.978613, 0.979240, 0.981272]


# Plotting
plt.plot(img_size, acc, marker='o')
plt.title('Accuracy of models on large images by analysing multiple parts')
plt.xlabel('Image size used to train model')
plt.ylabel('Accuracy')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/Part3/size_vs_performance_plot2.png')

# Display the plot (optional)
plt.show()


plt.clf()


#PART1 PLOTS 2
loss = [0.220994, 0.096252, 0.019534, 0.016231, 0.012538, 0.012905, 0.012467, 0.011044]



# Plotting
plt.plot(img_size, loss, marker='o')
plt.title('Final Validation Loss for various sized images')
plt.xlabel('Image size')
plt.ylabel('Validation Loss')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('./results/Part1/size_vs_loss_plot2.png')

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