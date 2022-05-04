# Will Prototype Methods on Stock Market Data -- Data input doesn't need to be rigorously vetted (Proof of Concept Currently)
import pandas as pd 
from numpy import array
import numpy as np 


# UDFs

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
def getAutoencoderData(data_path, n_days=20, column="volume", ticker="AAPL", n_samples=10000):
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

	# Split Sequence into Randomized Windows
	X = random_sequence(sequence=input_sequence, n_steps=n_days, n_samples=n_samples, labels=False)

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


