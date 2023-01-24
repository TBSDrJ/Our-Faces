import numpy as np
import matplotlib.pyplot as plt

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required = True,
    help = "File containing log of epoch results.")
ap.add_argument("-e", "--epochs", required = False,
    help = "Number of epochs to use in graph, default is to use all.")
args = vars(ap.parse_args())
filename = args["file"]

tr_ls = []
tr_ac = []
vl_ls = []
vl_ac = []
with open(filename, 'r') as input_file:
    for line in input_file.readlines():
        if 'loss:' in line:
            parts = line.split()
            if float(parts[5]) < 2:
                tr_ls.append(float(parts[5])/2)
            else:
                tr_ls.append(1.)
            tr_ac.append(float(parts[8]))
            if float(parts[11]) < 2:
                vl_ls.append(float(parts[11])/2)
            else:
                vl_ls.append(1.)
            vl_ac.append(float(parts[14]))

if args["epochs"] is not None:
    epochs = int(args["epochs"])
else:
    epochs = len(tr_ls)

x = np.arange(epochs)

plt.plot(x, tr_ls[:epochs], color = 'red')
plt.plot(x, tr_ac[:epochs], color = 'orange')
plt.plot(x, vl_ls[:epochs], color = 'blue')
plt.plot(x, vl_ac[:epochs], color = 'green')
plt.show()
