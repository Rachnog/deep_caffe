import mnist_loader
import network2
import cifar_loader


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
validation_data = validation_data[:1000]
input_size = 784


'''
X_train, y_train, X_test, y_test = cifar_loader.load_data_for_nn()
input_size = 32*32*3

X_test = X_test[:100]
y_test = y_test[:100]

training_data = zip(X_train, y_train)
test_data = zip(X_test, y_test)
'''

net = network2.Network([input_size, 30, 10])
net.default_weight_initializer()
net.SGD(training_data, 30, 10, 0.8, lmbda = 3.0, evaluation_data = test_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)
