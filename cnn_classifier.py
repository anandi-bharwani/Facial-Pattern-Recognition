#Best error rate - 0.1

import numpy as np 
import theano.tensor as T 
import theano
#import matplotlib.pyplot as plt 
from util import init_weights, error_rate, load_data, y2indicator, get_4d_data
from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
import theano.tensor.signal.pool as pool

def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
	return w.astype(np.float32)

class ConvPoolLayer(object):
	def __init__(self, Mi, Mo, fw=5, fh=5, poolsz=(2,2)):
		sz = (Mo, Mi, fw, fh)
		W = init_filter(sz, poolsz)
		b = np.zeros(Mo).astype(np.float32)
		self.W = theano.shared(W)
		self.b = theano.shared(b)
		self.poolsz = poolsz
		self.params = [self.W, self.b]

	def forward(self, X):
		conv = conv2d(X,self.W)						#Convolution
		max_pool = pool.pool_2d(conv, ws=self.poolsz, ignore_border=True)	#Max-pooling
		return T.nnet.relu(max_pool + self.b.dimshuffle('x', 0, 'x', 'x'))		#Non-linearity

class HiddenLayer(object):
  def __init__(self, M1, M2):
    self.M1 = M1
    self.M2 = M2
    W = init_weights(M1, M2)
    b = np.zeros(M2).astype(np.float32)
    self.W = theano.shared(W, 'W')
    self.b = theano.shared(b, 'b')
    self.params = [self.W, self.b]
    
  def forward(self, X):
  	return T.nnet.relu(X.dot(self.W) + self.b)
    
class CNN(object):
  def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.convpool_layer_sizes = convpool_layer_sizes
    
  def forward(self, X):
    z = X
    for c in self.convpool_layers:
      z = c.forward(z)
    z = z.flatten(ndim=2)
    for h in self.hidden_layers:
      z = h.forward(z)
    return T.nnet.softmax(z.dot(self.W) + self.b)
    
  def fit(self, X, Y, lr=10e-7, mu=0.99, batch_sz=100):
    Y = Y.astype(np.int32)
    X, Y = shuffle(X,Y)
    N, c, d, d = X.shape
    print(len(Y))
    K = Y.shape[1]
    
    mu = np.float32(mu)
    lr = np.float32(lr)
    print("N:", N, "K:", K)

    #Create the convolution-pooling layers
    self.convpool_layers=[]
    mi = c
    outw = d
    outh = d
    for mo, fw, fh in self.convpool_layer_sizes:
      c = ConvPoolLayer(mi, mo, fw, fh)
      self.convpool_layers.append(c)
      outw = (outw - fw +1)/ 2
      outh = (outh - fh +1)/ 2
      mi = mo
 
    #Create the hidden layers
    self.hidden_layers = []
    m1 = int(self.convpool_layer_sizes[-1][0]*outw*outh)
    for m2 in self.hidden_layer_sizes:
      h = HiddenLayer(m1, m2)
      self.hidden_layers.append(h)
      m1 = m2 
      
    W = init_weights(m2, K)    #Logistic reg layer
    b = np.zeros([K]).astype(np.float32)
    
    #Create theano variables
    thX = T.tensor4('X', dtype='float32')
    thY = T.fmatrix('Y')
    
    self.W = theano.shared(W, 'W_log')
    self.b = theano.shared(b, 'b_log')
    
    #Create parameter array for updates
    params = [self.W, self.b]
    for c in self.convpool_layers:
      params += c.params
    for h in self.hidden_layers:
      params += h.params
    
    #Momentum parameters
    dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in params]

    #Forward pass
    pY = self.forward(thX)
    P = T.argmax(pY, axis=1)
    cost = -(thY * T.log(pY)).sum()

    #Weight updates
    updates = [
    		(p, p + mu*d - lr*T.grad(cost, p)) for p,d in zip(params, dparams)
   		] + [
    		(d, mu*d - lr*T.grad(cost, p)) for p,d in zip(params, dparams)
    	]
    #Theano function for training and predicting and calculating cost
    train = theano.function(
        inputs=[thX, thY],
        updates=updates,
        allow_input_downcast=True
      )
      
    get_cost_prediction = theano.function(
        inputs=[thX, thY],
        outputs=[P, cost],
        allow_input_downcast=True
      )
    
    #Loop for Batch grad descent
    no_batches = int(N/batch_sz)
    for i in range(500):
      #lr *= 0.9
      for n in range(no_batches):
        Xbatch = X[n*batch_sz:(n*batch_sz+batch_sz)]
        Ybatch = Y[n*batch_sz:(n*batch_sz+batch_sz)]
        #print(Xbatch.shape, Ybatch.shape)
        train(Xbatch, Ybatch)
        if n%100==0:
          Yb = np.argmax(Ybatch, axis =1)
          P, c = get_cost_prediction(Xbatch, Ybatch)
          #print(P.shape, Ybatch.shape)
          er = error_rate(P, Yb)
          print("iteration:", i, "cost:", c, "error rate:", er)


def main():
  X, Y = get_4d_data()
  print(X.shape) 
  model = CNN(
	convpool_layer_sizes=[(20,5,5), (20,5,5)],
	hidden_layer_sizes=[1000, 500],
	)
  model.fit(X, Y)


main()


