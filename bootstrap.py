import numpy as np
import matplotlib.pyplot as plt
import argparse

# Written by J. McCarty

# parse command line arguments
parser = argparse.ArgumentParser(
	description="Perform bootstrap analysis for histogram error bars"
)
parser.add_argument(
	"-f", required=True, help="Path to the raw data file."
)
parser.add_argument(
	"-GRID_MIN",type=float,required=True,help='GRID_MIN value'
)
parser.add_argument(
	"-GRID_MAX",type=float,required=True,help='GRID_MAX value'
)
parser.add_argument(
	"-NBINS",type=int,default=20,help='NBINS value'
)
parser.add_argument(
	"-column", type=int, default=1, help="Column index (starts with 0) to read from each file (default is 1 for second column)."
)
args = parser.parse_args()
# everything should be parsed by now
print("Will read column ",args.column+1, " from the data file")
# read distances (here we are reading distances from column 2)
datavalues = np.loadtxt(args.f, comments='#!',usecols=(args.column,))

print("histogram grid is ...\n")
print("GRID_MIN: ",args.GRID_MIN)
print("GRID_MAX: ",args.GRID_MAX)
print("NBINS: ",args.NBINS)
# get histogram bins or set fixed number of bins
# bins = 80

# automate bin selection
#bins = np.histogram_bin_edges(datavalues,bins='fd')
#print(bins)
#print(len(bins))
# manually set bins to be consistent with PLUMED
GRID_MIN=args.GRID_MIN
GRID_MAX=args.GRID_MAX
NBINS=args.NBINS
bins = np.linspace(GRID_MIN,GRID_MAX, NBINS+1)
#print(bins)
#print(len(bins))


hist, bins = np.histogram(datavalues, bins=bins, density=False)
#area = np.sum(hist*np.diff(bins))
#print(area)
hist = hist / np.sum(hist)
print('checking histogram normalization')
print('sum of probabilities is',np.sum(hist))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
#print(bin_centers)
#print(len(bin_centers))

# Bootstrap to find confidence bands on the histogram
n = datavalues.size
n_bootstrap = 1000 # this is the number of bootstraps to perform.
boot_hist = np.zeros((n_bootstrap, hist.size))
rng = np.random.default_rng(19767)  # this is a random number seed for reproducibility

for i in range(n_bootstrap):
	sample = rng.choice(datavalues, size=n, replace=True) # sampling with replacement
	boot_hist[i], _ = np.histogram(sample, bins=bins, density=False) # generating new histogram
	boot_hist[i] = boot_hist[i] / np.sum(boot_hist[i])

# confidence intervals (95% interval lies between the 2.5th percentile and 97.5th percentile)
lower, upper = np.percentile(boot_hist, [2.5, 97.5], axis=0)


plt.plot(bin_centers, hist)
plt.fill_between(bin_centers, lower, upper, alpha=0.3, label='95% bootstrap CI')
#plt.step(bin_centers,hist, where='post')
plt.ylabel("Probability")
plt.xlabel("distance [nm]")
plt.show()

# writing output here
outputfile = "histogram_with_CI.dat"
header = "# bin center probability_density lower_CI upper CI\n"

np.savetxt(outputfile,np.column_stack((bin_centers, hist, lower, upper)),header=header, fmt="%.6f")
