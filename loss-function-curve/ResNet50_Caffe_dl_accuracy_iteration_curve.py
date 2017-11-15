import os
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import pandas as pd
import matplotlib.pyplot as plt

# display plots in this notebook
# %matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

log_path = "/home/ucla/Downloads/Log-output/"
# log_path = "/home/jenifferwu/IMAGE_MASKS_DATA/Log-output/"
# loss_function_curve_data = os.path.join(log_path, "loss_function_curve_data.csv")
loss_function_curve_data = os.path.join(log_path, "accuracy1_curve_data_ResNet50.csv")

headers = ['loss_time', 'test_net_accuracy1_data']
df = pd.read_csv(loss_function_curve_data, names=headers)
print (df)

loss_time = df['loss_time']
test_net_accuracy1 = df['test_net_accuracy1_data'].astype(float)

x = range(len(loss_time))
y_1 = test_net_accuracy1
new_x = x[50:]
new_y_1 = y_1[50:]

# plot
# Create two subplots sharing y axis
# fig, (ax1, ax2) = plt.subplots(2, sharey=True)
fig, (ax2) = plt.subplots(1, sharey=True)

# ax1.plot(new_x, new_y_1, 'C1')
# ax1.set(title='Lung nodule\'s Caffe deep learning loss', ylabel='Test net loss')
# ax1.set(title='ResNet50 Caffe deep learning loss', ylabel='Test net loss')

ax2.plot(new_x, new_y_1, 'C1')
ax2.set(title='ResNet50 Caffe deep learning Accuracy1', xlabel='Iteration', ylabel='Train net Accuracy1')

plt.show()