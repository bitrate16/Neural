import matplotlib.pyplot as plt
import sys

X, Y = [], []

if len(sys.argv) < 2:
	print 'Filename expected'
	sys.exit(0)
	
# Open from argv[1]
file = open(sys.argv[1], 'r')
# Read 1st line with amount of pairs
file.readline()

for line in file:
	values = [float(s) for s in line.split()]
	X.append(values[0])
	Y.append(values[1])

plt.plot(X, Y, 'ro')
plt.show()

# Does plotting specified neural test set
# python plot_linear_set.py