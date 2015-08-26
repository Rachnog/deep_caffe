import cifar_loader
import network2

training_data, test_data = cifar_loader.load_hog()
validation_data = test_data[:100]
input_size = 32*32


net = network2.Network([input_size, 50, 10])
net.default_weight_initializer()
net.SGD(training_data, 20, 10, 0.5, lmbda = 10.0, evaluation_data = validation_data, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)


