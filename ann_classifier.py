#Best error rate - 0.1

import numpy as np 
import theano.tensor as T 
import theano
#import matplotlib.pyplot as plt 
from util import init_weights, error_rate, load_data, y2indicator, get_image_data

  
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
    
class ANN(object):
  def __init__(self, hidden_layer_sizes):
    self.hidden_layer_sizes = hidden_layer_sizes
    
  def forward(self, X):
    z = X
    for h in self.hidden_layers:
      z = h.forward(z)
    return T.nnet.softmax(z.dot(self.W) + self.b)
    
  def fit(self, X, Y, lr=10e-7, mu=0.99, batch_sz=100):
    Y = Y.astype(np.int32)
    N, D = X.shape
    print(len(Y))
    K = Y.shape[1]
    
    mu = np.float32(mu)
    lr = np.float32(lr)
    print("D:", D, "N:", N, "K:", K)

    #Create the hidden layers
    self.hidden_layers = []
    m1 = D
    for m2 in self.hidden_layer_sizes:
      h = HiddenLayer(m1, m2)
      self.hidden_layers.append(h)
      m1 = m2 
      
    W = init_weights(m2, K)    #Logistic reg layer
    b = np.zeros([K]).astype(np.float32)
    
    #Create theano variables
    thX = T.fmatrix('X')
    thY = T.matrix('Y')
    
    self.W = theano.shared(W, 'W_log')
    self.b = theano.shared(b, 'b_log')
    
    #Create parameter array for updates
    params = [self.W, self.b]
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
  X, Y = load_data()
  
  model = ANN([1000, 500])
  model.fit(X, Y)


main()


