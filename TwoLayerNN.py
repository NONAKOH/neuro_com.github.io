import numpy as np
from mnist import load_mnist
from collections import OrderedDict
from matplotlib import pyplot

sin_train_accuracy = []
sin_test_accuracy = [0]

learning_list = []
max_train_list = []
max_test_list = []

for n in range(23):
    class TwoLayerNet:

        def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
            self.params = {}
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
            self.params['b2'] = np.zeros(output_size)
            self.layers = OrderedDict()
            self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
            self.layers['Sigmoid'] = Sigmoid()    #シグモイド関数を使用する場合
            #self.layers['Relu1'] = Relu()          #ReLU関数を使用する場合
            self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
            self.lastLayer = SoftmaxWithLoss()
            
        def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)
            return x
            
        def loss(self, x, t):
            y = self.predict(x)
            return self.lastLayer.forward(y, t)
        
        def accuracy(self, x, t):
            y = self.predict(x)
            y = np.argmax(y, axis=1)
            if t.ndim != 1 : t = np.argmax(t, axis=1)
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy
            
        def numerical_gradient(self, x, t):
            loss_W = lambda W: self.loss(x, t)
            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
            return grads
            
        def gradient(self, x, t):
            # forward
            self.loss(x, t)
            # backward
            dout = 1
            dout = self.lastLayer.backward(dout)
            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)
            grads = {}
            grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
            grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
            return grads

    class Affine:
        def __init__(self, W, b):
            self.W =W
            self.b = b
            self.x = None
            self.original_x_shape = None
            self.dW = None
            self.db = None

        def forward(self, x):
            self.original_x_shape = x.shape
            x = x.reshape(x.shape[0], -1)
            self.x = x
            out = np.dot(self.x, self.W) + self.b
            return out

        def backward(self, dout):
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis=0)
            dx = dx.reshape(*self.original_x_shape)
            return dx

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    class Sigmoid:
        def __init__(self):
            self.out = None

        def forward(self, x):
            out = sigmoid(x)
            self.out = out
            return out

        def backward(self, dout):
            dx = dout * (1.0 - self.out) * self.out
            return dx

    class Relu:
        def __init__(self):
            self.mask = None

        def forward(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            out[self.mask] = 0
            return out

        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout
            return dx
        
    def softmax(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


    class SoftmaxWithLoss:
        def __init__(self):
            self.loss = None
            self.y = None
            self.t = None

        def forward(self, x, t):
            self.t = t
            self.y = softmax(x)
            self.loss = cross_entropy_error(self.y, self.t)
            
            return self.loss

        def backward(self, dout=1):
            batch_size = self.t.shape[0]
            if self.t.size == self.y.size:
                dx = (self.y - self.t) / batch_size
            else:
                dx = self.y.copy()
                dx[np.arange(batch_size), self.t] -= 1
                dx = dx / batch_size
            
            return dx

    def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    max_epochs = 30
    train_size = x_train.shape[0]
    batch_size = 100

    learning_rate = 0.1 + 0.02*n      #学習率
    learning_rate=np.round(learning_rate, decimals=2)
    learning_list.append(learning_rate)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    epoch_list = []
    train_accuracy = []
    test_accuracy = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(10000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            #print("Epoch:" + str(epoch_cnt) + ", accuracy (train data):" + str(train_acc) + ", accuracy (test data):" + str(test_acc))
            epoch_list.append(epoch_cnt)
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            
            epoch_cnt += 1
            if epoch_cnt > max_epochs:
                break

    max_train_list.append(np.max(train_accuracy))
    max_test_list.append(np.max(test_accuracy))

    if np.max(test_accuracy) >= np.max(sin_test_accuracy):
        sin_train_accuracy = train_accuracy
        sin_test_accuracy = test_accuracy
        sin_learning_rate = learning_rate
    
    print(learning_rate)

print("max accuracy learning rate:" + str(sin_learning_rate) + ", test accuracy:" + str(sin_test_accuracy))

#学習率　図の描画
pyplot.title("model accuracy(learning rate:" + str(sin_learning_rate) + ")", {"fontsize":25})
pyplot.xlabel("learning rate", {"fontsize":10})
pyplot.ylabel("accuracy", {"fontsize":10})
pyplot.plot(learning_list,max_train_list, label='train_max')
pyplot.plot(learning_list,max_test_list, label='test_max')
pyplot.legend()
pyplot.show()

#図の描画
pyplot.title("model accuracy", {"fontsize":25})
pyplot.xlabel("epoch", {"fontsize":10})
pyplot.ylabel("accuracy", {"fontsize":10})
pyplot.plot(epoch_list,sin_train_accuracy, label='train')
pyplot.plot(epoch_list,sin_test_accuracy, label='test')
pyplot.legend()
pyplot.show()


