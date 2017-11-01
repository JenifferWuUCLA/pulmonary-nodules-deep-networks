import os
import argparse
from matplotlib import pyplot as plt
import numpy as np


class Out_put(object):
    def __init__(self, test_iter, max_iter, display, train_loss, test_accu):
        self.test_iter = test_iter
        self.max_iter = max_iter
        self.display = display
        self.train_loss = train_loss
        self.test_accu = test_accu


class Loss(object):
    def __init__(self, log, out_put):
        self.log = log
        self.out_put = out_put

    def draw_loss(self):
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(np.arange(0, self.out_put.max_iter + self.out_put.display, self.out_put.display),
                 self.out_put.train_loss)
        ax2.plot(self.out_put.test_iter * np.arange(len(self.out_put.test_accu)), self.out_put.test_accu, "r")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("train loss")
        ax2.set_ylabel("test accuracy")
        plt.show()

    def load_loss(self):
        log_file = open(self.log)
        for line in open(self.log):
            line = log_file.readline()
            line = line.strip()
            flag = line.find("test_interval")
            if flag >= 0:
                self.out_put.test_iter = int(line[14:])

            flag = line.find("max_iter")
            if flag >= 0:
                self.out_put.max_iter = int(line[10:])

            flag = line.find("display")
            if flag >= 0:
                self.out_put.display = int(line[8:])

            flag = line.find("Test net output")
            if flag >= 0:
                flag = line.find("Accuracy1")
                if flag >= 0:
                    self.out_put.test_accu.append(float(line[flag + 12:]))

            flag = line.find("Iteration")
            if flag >= 0:
                flag = line.find("loss")
                if flag >= 0:
                    self.out_put.train_loss.append(float(line[flag + 6:]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.log == None or (not os.path.exists(args.log)):
        print "check logfile"
        exit(0)
    out_put = Out_put(None, None, None, [], [])
    loss = Loss(args.log, out_put)
    loss.load_loss()
    loss.draw_loss()
