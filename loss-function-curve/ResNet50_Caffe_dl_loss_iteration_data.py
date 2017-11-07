import csv, os

log_path = "/home/ucla/Downloads/Log-output/"
# log_path = "/home/jenifferwu/IMAGE_MASKS_DATA/Log-output/"
train_caffenet_lungnet_file = os.path.join(log_path, "train_caffenet_ResNet50.log")
loss_function_curve_data = os.path.join(log_path, "loss_function_curve_data_ResNet50.csv")

########################################################################################################################
lossRows = []


def loss_function_curve_row(loss_time, test_net_loss_data, train_net_loss_data):
    new_row = []
    new_row.append(loss_time)
    new_row.append(test_net_loss_data)
    new_row.append(train_net_loss_data)
    lossRows.append(new_row)


########################################################################################################################

# Read the train_caffenet_lungnet.log in.
log_Rows = []
csvFileObj = open(train_caffenet_lungnet_file)
readerObj = csv.reader(csvFileObj)
logFile = open(train_caffenet_lungnet_file)
logLines = logFile.readlines()
for row in logLines:
    log_Rows.append(row)
csvFileObj.close()

# loss_function_curve_row("loss_time", "Test net loss", "Train net loss")

for log_row in log_Rows:
    log_str = log_row
    print(log_str)
    if "218] Iteration " in log_str:
        t_sIndex = log_str.index("] Iteration ")
        t_eIndex = log_str.index(" (")
        iteration = log_str[t_sIndex + 12: t_eIndex].strip()
        print(iteration)
        print(log_str)
        l_sIndex = log_str.index("loss = ")
        loss = log_str[l_sIndex + 7:].strip()
        # print(loss)
        train_net_loss_data = loss
        for i in range(40):
            loss_function_curve_row(iteration, "0", train_net_loss_data)

# Write out the loss_function_curve_data.csv file.
print(loss_function_curve_data)
csvFileObj = open(loss_function_curve_data, 'w')
csvWriter = csv.writer(csvFileObj)
for row in lossRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()