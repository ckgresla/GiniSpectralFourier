# Convert Daily Volume Distribution (for all stocks) into Gini Coefficients before read in
import pandas as pd 
from numpy import array
import numpy as np 
from numpy.fft import fft, ifft


# UDFs

# GoFast Gini Calculation -- this CumSum version is 1000x faster than base RMAD calculation
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
	# Compute Without Weights -- Typical Usage
	else:
		x_sorted = np.sort(x)
		n = len(x)
		x_cum = np.cumsum(x_sorted, dtype=float)
		gc = (n + 1 - 2 * np.sum(x_cum) / x_cum[-1]) / n
	return gc


# Split Data into Sequences ('X' is the sequence up to the `y` date)
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# Find end of Current Sequence
		end_ix = i + n_steps
		# If Beyond end of whole data, Break
		if end_ix > len(sequence)-1:
			break
		# Define Input sequence and Output (being data and label)
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] #up to the last value in the sequence
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y) #np.arrays


# Random Split Variant of above Sequence Generator
# Would recommend doing up to 1,264 (leave last 20 values as holdout) -- Default behavior is to return labels
def random_sequence(sequence, n_steps, n_samples=5000, labels=True):
	X, y = list(), list()
	
	for i in range(n_samples):
		i = np.random.randint(low=0, high=len(sequence)-n_steps) #get random start index between 0th value and up to held out set (last 20 days if n_steps=20)
		# Calculate end of Current Window
		end_ix = i + n_steps

		# If Beyond end of whole data, Break -- might not be getting triggered/is this necessary?
		if end_ix > len(sequence)-1:
			break
		
		# Build Labeled or Unlabeled Arrays -- Could refactor below, get rid of X, y variant
		if labels:
			# Define Input sequence and Output (being data and label)
			seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] #up to the last value in the sequence
			# Append Instances
			X.append(seq_x)
			y.append(seq_y)
		else:
			seq_x = sequence[i:end_ix]
			X.append(seq_x)

	if labels:
			return array(X), array(y) #return sequences as np.arrays
	else:
			return array(X) #returns Xs for Autoencoder (no need to labeled set)


# Reads Data & Supplies ready to go train & test sets -- Taking Alt Approach in `getAutoencoderData`
def getData(data_path, n_days=20, column="volume", ticker="AAPL", n_samples=10000):
	# Read in Data & Subset
	df = pd.read_csv(data_path)
	df.rename(columns={"Name":"ticker"}, inplace=True)
	df["date"] = [pd.Timestamp(x) for x in df["date"]]

	# Subset Data for 1 Ticker
	dfs = df[df["ticker"] == ticker]


	# Input Sequence for Windowing -- df is sorted already, so values are in increasing time order (can take last few values as test set)
	input_sequence = dfs[column].to_list() #currently set to Volume -- can be any numeric in df.columns

	# N-Time Steps per Window
	n_days = n_days #20 days = 4 weeks if 1 month windows (of business days)

	# Split Sequence into Windows
	#X, y = split_sequence(input_sequence, n_steps=n_days) #gets data in arrays -- original variant, unrandomized
	X, y = random_sequence(input_sequence, n_steps=n_days, n_samples=n_samples) #returns randomly generated sequences

	# Fill All Zeroes with Prior Value
	while True:
		I = np.nonzero(X==0)[0]
		if len(I)==0: break
		X[I] = X[I-1]
	
	# # Return Either Normalized or UnScaled Data -- previously did normalization in func read in, will do in nb to get reverse-transforms
	# if normalize:
	# 	# normalize the data
	# 	scaler = MinMaxScaler() #get between 0 & 1
	# 	y = y.reshape(-1, 1)
	# 	X, y = scaler.fit_transform(X), scaler.fit_transform(y)

	# Reshape Dims of Data for LSTM & Split into Train/Test Sets (test set is last portion of data)
	# test_set_size = len(X) - 284 #specify test set size and split accordingly -- did manually originally
	test_set_size = n_days #len of test set is the size of sequence
	#X = X.reshape((len(X), n_days, 1)) #reshapes to- (1284, 20, 1) = [samples, timesteps_per_sample, n_features in timestep] -- ORIGINAL SHAPE, diff below
	X = X.reshape((len(X), 1, n_days)) #reshapes to- (1284, 1, 20) = [samples, timestep, features (20 values per timestep)] -- ALT SHAPE, solved
		# main difference for above change: at any given timestep, we feed the model 20 "features" which happen to be the previous 20 volume vals
	X_train, y_train =  X[test_set_size:], y[test_set_size:]
	X_test, y_test = X[:test_set_size], y[:test_set_size]

	# Print dims of Train & Test Sets
	print("Train - Data-{} Labels-{}".format(X_train.shape, y_train.shape))
	print(" Test - Data-{} Labels-{}".format(X_test.shape, y_test.shape))
	return X_train, y_train, X_test, y_test #return arrays split nicely (can scale later)


# Reads historical data for plotting at end? 
def getHistoricalData(data_path, n_days=20, column="volume", ticker="AAPL", normalize=False):
	# Read in Data & Subset
	df = pd.read_csv(data_path)
	df.rename(columns={"Name":"ticker"}, inplace=True)
	df["date"] = [pd.Timestamp(x) for x in df["date"]]

	# Subset Data for 1 Ticker
	dfs = df[df["ticker"] == ticker]

	# Input Sequence for Windowing -- df is sorted already, so values are in increasing time order (can take last few values as test set)
	input_sequence = dfs[column].to_list() #currently set to Volume -- can be any numeric in df.columns

	# N-Time Steps per Window
	n_days = n_days #20 days = 4 weeks if 1 month windows (of business days)

	# Split Sequence into Windows
	X, y = split_sequence(input_sequence, n_days) #gets data in arrays -- original variant, unrandomized
	

	# Fill All Zeroes with Prior Value
	while True:
		I = np.nonzero(X==0)[0]
		if len(I)==0: break
		X[I] = X[I-1]

	# Reshape Dims of Data for LSTM & Split into Train/Test Sets
	#test_set_size = len(X) - 284 #specify test set size and split accordingly
	#X = X.reshape((len(X), n_days, 1)) #reshapes to- (1284, 20, 1) = [samples, timesteps_per_sample, n_features in timestep]
	X = X.reshape((len(X), 1, n_days)) #shape for LSTM

	# Print dims of Train & Test Sets
	print("Hist. - Data-{} Labels-{}".format(X.shape, y.shape))

	return X, y #need scaler for unscaling later 
	

# Alt approach for getting random sequences of data (did not want remove label version in `getData` although autoencoders do not need labels)
def getAutoencoderGiniData(data_path, n_days=20, column="volume", n_samples=10000):
	# Read in Data & Subset
	df = pd.read_csv(data_path)
	df.rename(columns={"Name":"ticker"}, inplace=True)
	df["date"] = [pd.Timestamp(x) for x in df["date"]]

	# Subset Data for 1 Ticker
	dfs = df #no longer subsetting, using categorical gini dist

	# Input Sequence for Windowing -- df is sorted already, so values are in increasing time order (can take last few values as test set)
	# input_sequence = dfs[column].to_list() #currently set to Volume -- can be any numeric in df.columns
	dfs = dfs[[column, "date", "ticker"]] #need 3 columns to get daily ginis
	dfs.sort_values(by="date", inplace=True)
	dfs = calculateGinis(dfs, datetime="date", numeric=column, return_df=True, impute_prior_day=True)
	input_sequence = dfs["GINI"] #list of ordered (by date) Gini Coefficients

	# N-Time Steps per Window
	n_days = n_days #20 days = 4 weeks if 1 month windows (of business days)

	# Split Sequence into Randomized Windows of Gini Coefficients
	# X = random_sequence(sequence=input_sequence, n_steps=n_days, n_samples=n_samples, labels=False) #before Ginis
	X = random_ginis(sequence=input_sequence, n_steps=n_days, n_samples=n_samples) #with Ginis

	# Fill All Zeroes with Prior Value
	while True:
		I = np.nonzero(X==0)[0]
		if len(I)==0: break
		X[I] = X[I-1]

	# Reshape Dims of Data for LSTM & Split into Train/Test Sets (test set is last portion of data)
	test_set_size = n_days #len of test set is the size of sequence
	#X = X.reshape((len(X), n_days, 1)) #reshapes to- (1284, 20, 1) = [samples, timesteps_per_sample, n_features in timestep] -- ORIGINAL SHAPE, diff below
	X = X.reshape((len(X), 1, n_days)) #reshapes to- (1284, 1, 20) = [samples, timestep, features (20 values per timestep)] -- ALT SHAPE, solved
		# main difference for above change: at any given timestep, we feed the model 20 "features" which also happen to be the previous 20 volume vals

	X_train =  X[test_set_size:]
	X_test = X[:test_set_size]

	# Print dims of Train & Test Sets -- Autoencoder has no label
	print("Train - Data-{}".format(X_train.shape))
	print(" Test - Data-{}".format(X_test.shape))
	return X_train, X_test


# Get Randomized Autoencoder Data of Gini Coefficients
def random_ginis(sequence, n_steps, n_samples=5000):
	X, y = list(), list()

	for i in range(n_samples):
		i = np.random.randint(low=0, high=len(sequence)-n_steps) #get random start index between 0th value and up to held out set (last 20 days if n_steps=20)
		# Calculate end of Current Window
		end_ix = i + n_steps

		# If Beyond end of whole data, Break -- might not be getting triggered/is this necessary?
		if end_ix > len(sequence)-1:
			break
		
		seq_x = sequence[i:end_ix]
		X.append(seq_x)
		
	return array(X) #returns Xs for Autoencoder (no need to labeled set)


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
			gc = giniGoFast(d_distribution)

		# Append Gini to Dict
		gini_dict["GINI"].append(gc) #either imputed or clean Gini Val

	# Define Return Items -- Either Sorted DF or Raw Dictionary
	if return_df:
		gini_df = pd.DataFrame.from_dict(gini_dict)
		gini_df = gini_df.sort_values(by="DATE")
		return gini_df
	else:
		return gini_dict


# FFT Wrapper - can transform into signal domain or inverse back into time domain
def fourier_transform(x, inverse=False):
	if inverse:
		# Inverse Transform - Recreate Original Sequence
		x = np.abs(np.fft.ifft(x))
		# print("Inverse Transform\n", x[0], "\n")

	# Apply Regular Fourier Transform (transfer TO signal domain)
	else:
		x = fft(x, axis=2) #apply to distribution of Ginis (axis=2)
		amp = (x[:].real**2 + x[0][:].imag**2) #get sum of squares for complex nums
		amp_scaled = np.divide(amp, len(amp)) #reduced magnitude by number of instances (something like normalization)
		# print("Transformed\n", x[0], "\n")

	return x



