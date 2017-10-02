import numpy as np
import scipy.special
import sys
import matplotlib.pyplot as plt


class Model:
    length = 28
    width = 28
    grey_scale_min = 0
    grey_scale_max = 255
    bg_matrix = [0]
    std_bg = 125

    def __init__(self, length, width, grey_scale_min, grey_scale_max):
        self.length, self.width = length, width
        self.grey_scale_min, self.grey_scale_max = grey_scale_min, grey_scale_max
        self.bg_matrix = np.zeros((length, width))
        self.std_bg = (grey_scale_min + grey_scale_max) >> 1
        self.bg_matrix += self.std_bg

        print("init result:")
        print("length: {} \n width: {}\n background: {}".format(
            length, width, self.bg_matrix))
        pass

    def fit(self, direction, lr, data):  #ideal learning rate=1/m
        if (direction):
            diff = (data - self.std_bg) * lr
            pass
        else:
            diff = (data + self.std_bg) * lr
            pass
        bg = self.bg_matrix
        self.bg_matrix += diff
        #print("Train:\ndiff: {}".format(diff))
        pass

    def query(self, m_subject):
        m_subject -= self.std_bg
        diff = (m_subject - self.bg_matrix)
        diff = np.reshape(diff, -1)
        error = 0
        for i in diff:
            error += int(i)**2
            pass
        return (error / (self.length * self.width))**0.5

    pass


models = [
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255),
    Model(28, 28, 0, 255)
]

#load training set
training_data_file = open("../mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for train_data in training_data_list:
    label = train_data.split(',')[0]
    data = np.asfarray(train_data.split(',')[1:]).reshape(28, 28)
    for i in range(0, 10):
        if (label == str(i)):
            models[i].fit(True, 0.01, data)
            pass
        else:
            models[i].fit(True, 0.01, data)
            pass
        pass
    pass

for model in models:
    plt.imshow(model.bg_matrix, cmap='Greys', interpolation='None')
    #plt.show()
    pass

#load testing set
testing_data_file = open("../mnist_dataset/mnist_train.csv", 'r')
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

for test_data in testing_data_list:
    label = test_data.split(',')[0]
    data = np.asfarray(test_data.split(',')[1:]).reshape(28, 28)

    result=0
    results = []
    error = 99999999999999
    for i in range(0, 10):
        results.append(models[i].query(data))
        pass

    for i in range(0, 10):
        if(error>results[i]):
            result=i
            error=results[i]
            pass
        pass

    print(results)

    print("label: {}\nresult: {}\nerror:{}".format(label, result, error))
    plt.imshow(models[result].bg_matrix, cmap='Greys', interpolation='None')
    plt.show()
    plt.imshow(data, cmap='Greys', interpolation='None')
    plt.show()

    pass
