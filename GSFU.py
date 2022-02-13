
"""
GSFU = Gini Spectral Fourier Utils

This file contains a collection of generalized functions that cover the main steps taken when analyzing a new dataset. 
	1. Gini Functions: Function that calculates the gini coefficient by using half of the Relative Mean Absolute Difference formula (as opposed to the lorenz curve -- two implementations of varying speeds, fastest included in production code)
	2. Dates Sequence Checker: Exists to flag errors, ideally helps direct EDA (Returns the Uninterupted Time Sequence & all Missing Entries)
	3. Calculate Ginis: Applies the single Gini function (func 1) on a date-by-date basis for some array of numeric values (representing the distribution of some categorical)
	4. Plot Gini Distribution: Take output from Gini Calculation step and plot coefficients over a date range
	5. Transform Ginis & Plot Fourier: Map the Gini Calculation output to the frequency domain

can import below functions with;
from GSFU import *
"""


# Dependency Imports + Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

plt.style.use("seaborn-pastel")


# TimeFunc Decorator
import time
import functools

def timefunc(func):
    """timefunc's doc"""

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(f"FUNCTION: {func.__name__} | TIME: {time_elapsed}")
        return result
    return time_closure




#1 – Gini Calculation Function -- Using "Relative Mean Absolute Difference" Approach (Mathematically Equivalent to Classic Econ Method)
# Referenced Post- https://stackoverflow.com/questions/48981516/weighted-gini-coefficient-in-python
#@timefunc #decorated time function, use for testing
def gini(x, weights=None): 
	if weights is None: #can weight specific inputs, non-essential if we just pass in the weights themselves (volume in this case)
		weights = np.ones_like(x)
	count = np.multiply.outer(weights, weights)
	mad = np.abs(np.subtract.outer(x, x) * count).sum() / count.sum() #mean absolute deviation
	rmad = mad / np.average(x, weights=weights) #relative mean absolute deviation 
	
	return 0.5 * rmad #half the RMAD is mathematically equivalent to the Lorenz Curve used in typical Economics Gini Calculations


# GoFast Gini -- 3 orders of magnitude faster than original
#@timefunc 
def giniGoFast(x, w=None):
	"""
	Reference the StackOverflow Article, need make gini function faster
	Post- https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
	"""
	# Compute With Weights
	if w is not None:
		w = np.asarray(w)
		sorted_indices = np.argsort(x)
		x_sorted = x[sorted_indices]
		w_sorted = w[sorted_indices]
		# Force float dtype to avoid overflows
		cumw = np.cumsum(w_sorted, dtype=float)
		cumxw = np.cumsum(x_sorted * w_sorted, dtype=float)
		gc = (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / (cumxw[-1] * cumw[-1]))
	# Compute Without Weights
	else:
		x_sorted = np.sort(x)
		n = len(x)
		x_cum = np.cumsum(x_sorted, dtype=float)
		gc = (n + 1 - 2 * np.sum(x_cum) / x_cum[-1]) / n
	return gc


# GoFaster? Gini -- Tested but ultimately was not a better solution than the goFast Function
#@timefunc #using %timeit
#def giniCumSumV2(x, w=None):
#	"""
#	Reference the StackOverflow Article, need make gini function faster
#	Post- https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
#	"""
#	# Array indexing requires reset indexes.
#	x = pd.Series(x).reset_index(drop=True)
#	if w is None:
#		w = np.ones_like(x)
#	w = pd.Series(w).reset_index(drop=True)
#	n = x.size
#	wxsum = sum(w * x)
#	wsum = sum(w)
#	sxw = np.argsort(x)
#	sx = x[sxw] * w[sxw]
#	sw = w[sxw]
#	pxi = np.cumsum(sx) / wxsum
#	pci = np.cumsum(sw) / wsum
#	g = 0.0
#	for i in np.arange(1, n):
#		gc = g + pxi.iloc[i] * pci.iloc[i - 1] - pci.iloc[i] * pxi.iloc[i - 1]
#	return gc


#2 – Date Sequence Checker -- check if dates array is interupted & print what dates are missing -- return an array of what dates should be?
def checkSequence(df, date, sequence_type=None, print_missing=False): #7-Day or 5-Day Type (full week or business version)
	
	# Get df Date Range Bounds & Instantiate List to Hold Missing Dates
	rng = df[date].unique()
	missing_dates = []
	num_msng = 0

	# Get UnInterrupted Sequence for df's Date Range
	if sequence_type == "7-Day":
		UninterruptedSequence = pd.date_range(start=rng.min(), end=rng.max()) #all dates valid
	elif sequence_type == "5-Day":
		UninterruptedSequence = pd.bdate_range(start=rng.min(), end=rng.max()) #only business dates valid (as defined by Pandas)
	else:
		print("No Standard Week Sequence Passed - ['7-Day', '5-Day']--> pass one of these values to 'sequence_type' variable in function")
		return


	# Print out Array-Level Info
	print(f"# of Dates in UninterruptedSequence: {len(UninterruptedSequence)}")
	print(f"       # of Dates in ActualSequence: {len(rng)}")

	# Print out missing dates in Real Sequence if Flag 'True'
	if print_missing:
		offset = 0 #keep track of missing dates
		for i in range(0, len(UninterruptedSequence)):
			try:
				true_date = UninterruptedSequence[i+offset]
			except:
				pass
			true_date = str(true_date)[:10] #format str entry correctly

			try:
				obsv_date = rng[i]
			except:
				num_msng+=1 
				# no longer any values in Data Date's Array
				pass
			
			if true_date != obsv_date:
				print(f"Error At Date: {true_date} | Observed Date: {obsv_date}")
				missing_dates.append(true_date)
				offset += 1
	
	print("\n# of Missing Dates:", num_msng)
	return UninterruptedSequence, missing_dates


#3 – Apply Gini Function to an Array of Numerics per Date in Date Array -- takes in uninterrupted DateTime Sequence & Numeric values as Input
def calculateGinis(df, datetime, numeric, return_df=True, impute_prior_day=True):
	"""
	Function to apply the Gini calculator to any date & numeric column in a Pandas Dataframe
	"""

	# Item to be Returned at End of Function
	gini_dict = {'DATE':[], "GINI":[]}

	# Calc Ginis for Each Unique Date
	for d in df[datetime].unique():

		# Filter df for Specified Unique Date
		dfd = df[df[datetime] == d]
		
		# Get Date & Daily Distribution/Gini Value
		gini_dict["DATE"].append(d)
		d_distribution = [val for val in dfd[numeric]] #distribution of categorical for current date

		#Save Previous Day Value if Imputing, Else Calc Gini Coefficient
		if impute_prior_day:
			try:
				# gc_1 is the Gini Value for DATE-1 (previous day's Gini Value or previous non-NaN value)
				gc_1 = gc #on first date in iteration, will not work as there is no prev GC to store
			except:
				pass

			# Calculate Gini Coefficient for Current Day
			#gc = gini(d_distribution) #original method with slow Gini Function
			gc = giniGoFast(d_distribution)

			# If IsNaN use prior day value
			if np.isnan(gc):
				gc = gc_1
		else:
			# Calculate Gini Coefficient for Current Day
			gc = gini(d_distribution)

		# Append Gini to Dict
		gini_dict["GINI"].append(gc) #either imputed or clean Gini Val

	# Define Return Items -- Either Sorted DF or Raw Dictionary
	if return_df:
		gini_df = pd.DataFrame.from_dict(gini_dict)
		gini_df = gini_df.sort_values(by="DATE")
		return gini_df
	else:
		return gini_dict


#4 – Plot Gini Coefficient Distribution over Time
def plotGinis(df, dates, ginis, line_color="#FF5671", x_label="Time (Days)", y_label="Gini Coefficients", title="Gini Coefficients over Time"):
	"""
	This func plots the gini coefficients calculated with "calculateGinis"

	Inputs must both be arrays or array-like objects (List, Pandas Column, Numpy array, etc.)
	DF MUST BE SORTED BY DATE ARRAY! (Gini Calculation Function returns sorted df)
	"""

	# Generate Plot
	plt.figure(figsize=(12, 8), dpi=800)
	ax = plt.axes()
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_title(title)
	ax.plot(dates, ginis, color=line_color)


#5 – Transform Ginis & Plot Fourier Values
def plotFourier(gini_array, mask_threshold=0.001, x_axis_range=(0.0, 0.5), line_color="#56B9FF", x_label="Frequencies", y_label="Amplitudes", title="Fourier Transformed Ginis | Frequency Domain"):
	"""
	Plot Fourier Transformed Gini Values, can specify the X axis range to zoom in on specific regions

	"""
	# Number of samples in `normalized_tone`
	N = len(gini_array) #input_array

	x = rfftfreq(N, d=1) #formula for freq is effectively f/N where f is the sample rate (d in this case) and N is the length of the input array (number of observations) -- d in this case essentially affects the length of the window (relating to the frequencies that the algorithm picks up)
	# Good resource for Developing Intuition for Frequency in FFT- http://www.baudline.com/
	y = rfft(gini_array) #calculate the Fourier Value on input array
	y = np.abs(y) #take absolute for plotting

	# Filter out the Noise (Signals very close to zero are likely to be noise, use the mask below to test different thresholds & the resulting graph)
	mask = (x >= mask_threshold) #change Float value to set cutoff threshold
	x = x[mask]
	y = y[mask]
	

	# Generate Plot
	plt.figure(figsize=(12, 8), dpi=800)
	plt.title(title)
	plt.xlim(x_axis_range)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.plot(x, y, color=line_color)


# 6 - Return Frequencies & Fourier Values, based on specified mask
def getFouriers(gini_array, mask_threshold=0):
	# Number of samples in `normalized_tone`
	N = len(gini_array) #input_array

	x = rfftfreq(N, d=1) #formula for freq is effectively f/N where f is the sample rate (d in this case) and N is the length of the input array (number of observations) -- d in this case essentially affects the length of the window (relating to the frequencies that the algorithm picks up)
	# Good resource for Developing Intuition for Frequency in FFT- http://www.baudline.com/
	y = rfft(gini_array) #calculate the Fourier Value on input array
	y = np.abs(y) #take absolute for plotting

	# Filter out the Noise (Signals very close to zero are likely to be noise, use the mask below to test different thresholds & the resulting graph)
	mask = (x >= mask_threshold) #change Float value to set cutoff threshold
	freqs = x[mask]
	fouriers = y[mask]
	
	return freqs, fouriers
