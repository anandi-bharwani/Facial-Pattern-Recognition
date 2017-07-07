import numpy as np

def error_rate(P, Y):
	return np.mean(P!=Y)

def init_weights(Mi, Mo):
	W = np.random.randn(Mi, Mo)/ np.sqrt(Mi + Mo)
	return W.astype(np.float32)

def y2indicator(Y, K):
    N = len(Y)
    Y_ind = np.zeros([N, K])
    Y = Y.astype(np.int32)
    for i in range(N):
        Y_ind[i, Y[i]] = 1
    return Y_ind

#Get fer.csv
def get_image_data():
	df = open('./fer2013/fer2013.csv')
	data = [s.strip() for s in df]

	X = np.zeros([len(data)-1, 48*48])
	Y = []
	for i in range(len(data) - 1):
		if i==0:
			pass
		else:
			row = data[i].split(',')
			Y.append(int(row[0]))
			X[i] = np.array([int(p)/255 for p in row[1].split()])
			#print(X)

	
	#X = np.array(X)
	Y = np.array(Y)
	K = len(set(Y))
	Y = y2indicator(Y, K)

	save_data(X, Y)
	print(X.shape, Y.shape)
	return X,Y

def save_data(X, Y):
	np.savez('fer_data.npz', X, Y)


#Since processing data takes too much time
def load_data():
	npzfile = np.load('fer_data.npz')
	X = npzfile['arr_0']
	Y = npzfile['arr_1']

	return X,Y

# print("Reading data")
# get_image_data()
# print("Loading data from file")
# X, Y = load_data()
# print(X.shape, Y.shape)
