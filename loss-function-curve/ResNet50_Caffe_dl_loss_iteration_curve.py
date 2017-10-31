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
loss_function_curve_data = os.path.join(log_path, "loss_function_curve_data_ResNet50.csv")

headers = ['loss_time', 'test_net_loss', 'train_net_loss']
df = pd.read_csv(loss_function_curve_data, names=headers)
print (df)

loss_time = df['loss_time']
test_net_loss = df['test_net_loss'].astype(float)
train_net_loss = df['train_net_loss'].astype(float)

x = range(len(loss_time))
y_1 = test_net_loss
y_2 = train_net_loss
# 去掉最初的50次迭代,zoom in看一下之后迭代的loss走势,现在的Loss function曲线被初始的较大loss遮盖掉了后续loss的走势.
new_x = x[50:]
new_y_1 = y_1[50:]
new_y_2 = y_2[50:]

# plot
# Create two subplots sharing y axis
# fig, (ax1, ax2) = plt.subplots(2, sharey=True)
fig, (ax2) = plt.subplots(1, sharey=True)

# ax1.plot(new_x, new_y_1, 'C1')
# ax1.set(title='Lung nodule\'s Caffe deep learning loss', ylabel='Test net loss')
# ax1.set(title='ResNet50 Caffe deep learning loss', ylabel='Test net loss')

ax2.plot(new_x, new_y_2, 'C2')
ax2.set(title='ResNet50 Caffe deep learning loss', xlabel='Iteration', ylabel='Train net loss')

plt.show()