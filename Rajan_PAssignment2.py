# Rajan, Rohin
# 1001-154-037

import kfoldValidation as kfold
import numpy as np
import math
import pandas
import operator

exception_column_headers = ["locsite"]


class ClEuclieanDistance:
    # function that calculates the distance with the train as well test dataset
    def calculate_distance(self, train_value, test_value):
        train_value.reshape(7, 1)
        test_value.reshape(7, 1)
        # multiplying in order to get the value square of the equation
        dist = pow(train_value - test_value, 2)
        # summing the equation in order to get the values of the function
        dist = np.sum(dist)
        return math.sqrt(dist)

    def getNeighbours(self, train_set, test_value, k):
        # iterate through the training set and a single test value
        distance = []
        # calculating the distances present in the data set and sorting it
        for train_value in train_set:
            new_train_value = train_value[0:len(train_value) - 1]
            new_test_value = test_value[0: len(test_value) - 1]
            dist = self.calculate_distance(new_train_value, new_test_value)
            # fetching the corresponding target values based on the train_set
            distance.append((train_value[-1], dist))
        # sorting the distances and fetching the
        distance.sort(key=operator.itemgetter(1))
        #       iterating through the top k list and getting the neighbours
        neighbours = []
        for indx in range(k):
            neighbours.append(distance[indx][0])
        return neighbours

    def get_count_of_neighbor(self, neigbhors):
        test_function1 = {}
        for nh in neigbhors:
            if test_function1.has_key(nh):
                test_function1[nh] += 1
            else:
                test_function1[nh] = 1
        return test_function1

    def get_max_neighbour(self, neighbours_dict):
        maxvalue = -1
        maxkey = None
        for key in neighbours_dict:
            if maxvalue < neighbours_dict[key]:
                maxvalue = neighbours_dict[key]
                maxkey = key
        return maxkey


class ClPolynomialKernel:

    # function to calculate the kernel for the polynomial kernel
    def kernel_calculation(self, train_value, test_value, p):
        train_value.reshape(1, 7)
        test_value.reshape(7, 1)
        kernel_value = np.dot(train_value, test_value)
        kernel_value += 1
        kernel_value = pow(kernel_value, p)
        return kernel_value

    def get_neighbours(self, train_set, test_value, p, k):
        distance = []
        # calculating the distances present in the data set and sorting it
        for train_value in train_set:
            new_train_value = train_value[0:len(train_value) - 1]
            new_test_value = test_value[0: len(test_value) - 1]
            dist = self.kernel_calculation(new_train_value, new_test_value, p)
            # fetching the corresponding target values based on the train_set
            distance.append((train_value[-1], dist))
        # sorting the distances and fetching the
        distance.sort(key=operator.itemgetter(1))
        #       iterating through the top k list and getting the neighbours
        neighbours = []
        for indx in range(k):
            neighbours.append(distance[indx][0])
        return neighbours

    def get_count_of_neighbor(self, neigbhors):
        test_function1 = {}
        for nh in neigbhors:
            if test_function1.has_key(nh):
                test_function1[nh] += 1
            else:
                test_function1[nh] = 1
        return test_function1

    def get_max_neighbour(self, neighbours_dict):
        maxvalue = -1
        maxkey = None
        for key in neighbours_dict:
            if maxvalue < neighbours_dict[key]:
                maxvalue = neighbours_dict[key]
                maxkey = key
        return maxkey


def calculate_accuracy(predicted_values, actual_values):
    total = 0
    for indx in range(len(predicted_values)):
        if predicted_values[indx] == actual_values[indx]:
            total += 1
    accuracy = (total / float(len(predicted_values))) * 100
    return round(accuracy, 2)


if __name__ == "__main__":
    dataset_path = "datasets/ecoli.csv"
    #  assuming the value k for k nn as well kfold validation
    k = 10

    train_set, test_set = kfold.k_fold_cross_validation(dataset_path, k, exception_column_headers)

    # converting the data sets into matrix values with dimenstions [8 * len(train or test set)]
    train_set = train_set.as_matrix().T
    test_set = test_set.as_matrix().T

    # # calculating the closest neighbors using euclidean distance
    # euclidean_dist = ClEuclieanDistance()
    # predicted_values = []
    # actual_values = []
    # for test_value in test_set.T:
    #     knn_neighbours = euclidean_dist.getNeighbours(train_set.T, test_value, k)
    #     nearest_neighbour = euclidean_dist.get_count_of_neighbor(knn_neighbours)
    #     predicted_values.append(euclidean_dist.get_max_neighbour(nearest_neighbour))
    #     actual_values.append(test_value[-1])
    #
    # print calculate_accuracy(predicted_values, actual_values)


    polynomial_dist = ClPolynomialKernel()
    predicted_values = []
    actual_values = []
    p = 4
    for test_value in test_set.T:
        knn_neighbours = polynomial_dist.get_neighbours(train_set.T,test_value,p,k)
        print knn_neighbours
        print test_value[-1]
        nearest_neighbour = polynomial_dist.get_count_of_neighbor(knn_neighbours)
        predicted_values.append(polynomial_dist.get_max_neighbour(nearest_neighbour))
        actual_values.append(test_value[-1])

    print calculate_accuracy(predicted_values, actual_values)

