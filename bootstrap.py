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
	"-column", type=int, default=1, help="Column index (starts with 0) to read from each file (default is 1 for second column)."
)
args = parser.parse_args()
# everything should be parsed by now
print("Will read column ",args.column+1, " from the data file")

# read distances (here we are reading distances from column 2) 
datavalues = np.loadtxt(args.f, comments='#!',usecols=(args.column,))

# get histogram bins or set fixed number of bins
# bins = 80
bins = np.histogram_bin_edges(datavalues,bins='fd')

hist, bins = np.histogram(datavalues, bins=bins, density=True)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Bootstrap to find confidence bands on the histogram 
n = datavalues.size 
n_bootstrap = 1000 # this is the number of bootstraps to perform.
boot_hist = np.zeros((n_bootstrap, hist.size))
rng = np.random.default_rng(19767)  # this is a random number seed for reproducibility 

for i in range(n_bootstrap):
	sample = rng.choice(datavalues, size=n, replace=True) # sampling with replacement
	boot_hist[i], _ = np.histogram(sample, bins=bins, density=True) # generating new histogram

# confidence intervals (95% interval lies between the 2.5th percentile and 97.5th percentile)
lower, upper = np.percentile(boot_hist, [2.5, 97.5], axis=0)


plt.plot(bin_centers, hist)
plt.fill_between(bin_centers, lower, upper, alpha=0.3, label='95% bootstrap CI')
plt.ylabel("Probability")
plt.xlabel("distance [nm]")
plt.show()


