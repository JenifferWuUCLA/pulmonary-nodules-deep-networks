import csv, os

log_path = "/home/ucla/Downloads/Log-output/"
# log_path = "/home/jenifferwu/IMAGE_MASKS_DATA/Log-output/"
train_caffenet_lungnet_file = os.path.join(log_path, "train_caffenet_lungnet.log")
loss_function_curve_data = os.path.join(log_path, "loss_function_curve_data.csv")

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
for row in readerObj:
    if readerObj.line_num == 1:
        continue  # skip first row
    log_Rows.append(row)
csvFileObj.close()

# loss_function_curve_row("loss_time", "Test net loss", "Train net loss")

for log_row in log_Rows:
    log_str = log_row[0]
    if "loss = " in log_str:
        t_sIndex = log_str.index("I0929 ")
        t_eIndex = log_str.index(" 29621")
        loss_time = log_str[t_sIndex + 1: t_eIndex].strip()
        l_sIndex = log_str.index("= ")
        l_eIndex = log_str.index(" (*")
        loss = log_str[l_sIndex + 1: l_eIndex].strip()
        # print(loss)
        if "Test net output" in log_str:
            test_net_loss_data = loss
            loss_function_curve_row(loss_time, test_net_loss_data, "0")
        elif "Train net output" in log_str:
            train_net_loss_data = loss
            loss_function_curve_row(loss_time, "0", train_net_loss_data)

# Write out the loss_function_curve_data.csv file.
print(loss_function_curve_data)
csvFileObj = open(loss_function_curve_data, 'w')
csvWriter = csv.writer(csvFileObj)
for row in lossRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()