#import numpy as np

class MatMul:
    def __init__(self,W):
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.x=None
        pass
    
    def forward(self, x):
        W, =self.params
        out=np.matmul(x,W)
        self.x=x
        return out

    def backward(self,dout):
        W, =self.params
        dx=np.matmul(dout,W.T)
        dW=np.matmul(self.x.T, dout)
        self.grads[0][...]=dW
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads=[],[]
        self.out=None
        pass

    def forward(self, x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out

    def backward(self, dout):
        dx=dout*(1.0-self.out)*self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.params=[W,b]
        self.grads=[np.zeros_like(W), np.zeros_like(b)]
        self.x=None
        pass

    def forward(self, x):
        W,b=self.params
        out=np.matmul(x,W)+b
        self.x=x
        return out

    def backward(self, dout):
        W, b=self.params
        dx=np.matmul(dout, W.T)
        dW=np.matmul(self.x.T, dout)
        db=np.sum(dout, axis=0)

        self.grads[0][...]=dW
        self.grads[1][...]=db
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads= [], []
        self.out=None
        pass 
    def forward(self, x):
        self.out=softmax(x)























