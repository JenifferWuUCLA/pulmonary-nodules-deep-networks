import csv, os

log_path = "/home/ucla/Downloads/Log-output/"
# log_path = "/home/jenifferwu/IMAGE_MASKS_DATA/Log-output/"
train_caffenet_lungnet_file = os.path.join(log_path, "train_caffenet_ResNet50.log")
accuracy1_curve_data = os.path.join(log_path, "accuracy1_curve_data_ResNet50.csv")

########################################################################################################################
accuracyRows = []


def accuracy1_curve_row(iteration, test_net_accuracy1_data):
    new_row = []
    new_row.append(iteration)
    new_row.append(test_net_accuracy1_data)
    accuracyRows.append(new_row)


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

# accuracy1_curve_row("iteration", "Test net Accuracy1")

iteration, test_net_accuracy1_data = "", ""
is_iteration = False
for log_row in log_Rows:
    log_str = log_row
    # print(log_str)

    if "330] Iteration " in log_str:
        t_sIndex = log_str.index("] Iteration ")
        t_eIndex = log_str.index(", ")
        iteration = log_str[t_sIndex + 12: t_eIndex].strip()
        print(iteration)
        # print(log_str)
        is_iteration = True

    if is_iteration and "Accuracy1 = " in log_str:
        l_sIndex = log_str.index("= ")
        accuracy1 = log_str[l_sIndex + 1:].strip()
        # print(accuracy1)
        test_net_accuracy1_data = accuracy1
        print(test_net_accuracy1_data)

        for i in range(35):
            if iteration != "" and test_net_accuracy1_data != "":
                print(iteration, test_net_accuracy1_data)
                accuracy1_curve_row(iteration, test_net_accuracy1_data)

        is_iteration = False

# Write out the accuracy1_curve_data.csv file.
print(accuracy1_curve_data)
csvFileObj = open(accuracy1_curve_data, 'w')
csvWriter = csv.writer(csvFileObj)
for row in accuracyRows:
    # print row
    csvWriter.writerow(row)
csvFileObj.close()