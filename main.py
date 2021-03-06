import math
import csv
import copy
import random
import datetime
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.3
MOMENTUM = 0.1
ACTIVATION_SLOPE_PARAM = 1
ACTIVATION_FUNCTION = "SIGMOID"
LAYERS = [8, 2]
NUM_LAYERS = len(LAYERS)
NUM_INPUT_FEATURES = 4
# the max number of epochs that will train if SSE does not converge below threshold
NUM_EPOCHS = 30
SSE_THRESHOLD = 0.001
NUM_FOLDS = 5


def main():

    weights = []
    biases = []

    total_predicted = []
    total_expected = []

    # randomly initalize our network weights
    for layer in range(NUM_LAYERS):
        layer_holder = []
        for node in range(LAYERS[layer]):
            if layer == 0:
                row = [random.uniform(-1, 1)
                       for _ in range(NUM_INPUT_FEATURES)]
            else:
                row = [random.uniform(-1, 1) for _ in range(LAYERS[layer-1])]
            layer_holder.append(row)
        weights.append(layer_holder)

    for layer in range(NUM_LAYERS):
        biases.append([random.uniform(-1, 1) for _ in range(LAYERS[layer])])

    # array to store SSE for each epoch
    sses = []

    prev_weights = []
    prev_biases = []

    # initialize empty training data array
    training_data_full = []

    # read training data
    # with open("data/training.csv", "r", encoding="utf-8-sig") as csvfile:
    #     contents = csv.reader(csvfile)
    #     for row in contents:
    #         new_row = [float(item) for item in row]
    #         training_data_full.append(new_row)

    # read data from txt file
    with open("data/training.txt", "r") as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            parts = line[1:-1].split("  ")
            i = 0
            tmp = []
            for part in parts:
                if i < NUM_INPUT_FEATURES:
                    tmp.append(float(part))
                else:
                    if part == '1':
                        tmp.append(1.0)
                        tmp.append(0.0)
                    else:
                        tmp.append(0.0)
                        tmp.append(1.0)
                i += 1
            training_data_full.append(tmp)

    start = datetime.datetime.now()

    training_folds = []
    for i in range(NUM_FOLDS):
        tmp = training_data_full[i*100:(i+1)*100] + \
            training_data_full[(5+i)*100:(5+i+1)*100]
        random.shuffle(tmp)
        training_folds.append(tmp)

    # training_folds = [training_data_full[i*(math.ceil(len(training_data_full)/NUM_FOLDS)):(
    #    i+1)*(math.ceil(len(training_data_full)/NUM_FOLDS))] for i in range(NUM_FOLDS)]

    for fold in range(NUM_FOLDS):
        prev_biases = []
        prev_weights = []

        weights = []
        biases = []

        sses = []

        training_data = []
        for i in range(NUM_FOLDS):
            if i != fold:
                training_data += training_folds[i]

        validation_data = training_folds[fold]

        # randomly initalize our network weights
        for layer in range(NUM_LAYERS):
            layer_holder = []
            for node in range(LAYERS[layer]):
                if layer == 0:
                    row = [random.uniform(-1, 1)
                           for _ in range(NUM_INPUT_FEATURES)]
                else:
                    row = [random.uniform(-1, 1)
                           for _ in range(LAYERS[layer-1])]
                layer_holder.append(row)
            weights.append(layer_holder)

        for layer in range(NUM_LAYERS):
            biases.append([random.uniform(-1, 1)
                           for _ in range(LAYERS[layer])])

        # loop for number of epochs
        for epoch in range(NUM_EPOCHS):

            # sum of squared errors for the epoch
            sse = 0

            # loop over pieces of data in training set
            for k in range(len(training_data)):

                # get the specific item out of the training set
                data = training_data[k]

                # split the item into its inputs and outputs
                inputs = data[:NUM_INPUT_FEATURES]
                expected_outputs = data[NUM_INPUT_FEATURES:]

                # no longer need the data row
                del data

                phi_v = []

                # loop over layers
                for layer in range(NUM_LAYERS):

                    # add placeholder for next layer
                    phi_v.append([])

                    # loop over nodes in layer
                    for node in range(len(weights[layer])):

                        # variable to track induced local field
                        v = 0

                        # if the layer is the first hidden layer, use inputs instead of y's
                        # from previous layer
                        if layer is 0:
                            for i in range(len(inputs)):
                                v += weights[layer][node][i] * inputs[i]

                        else:
                            # loop over the phi_v, or y, input to the node, updating v
                            for i in range(len(phi_v[layer-1])):
                                v += weights[layer][node][i] * \
                                    phi_v[layer-1][i]

                        # add the bias term for the node
                        v += biases[layer][node]

                        # store the phi(v) for the node
                        phi_v[layer].append(phi(v))

                # loop over output nodes
                errors = []
                for output_node in range(len(phi_v[NUM_LAYERS-1])):
                    errors.append(
                        expected_outputs[output_node] - phi_v[NUM_LAYERS-1][output_node])

                # calculate the contribution to SSE for this training item
                for error in errors:
                    sse += error ** 2

                deltas = [[] for _ in range(NUM_LAYERS)]
                # calculating deltas
                for layer in range(NUM_LAYERS-1, -1, -1):

                    # loop over nodes in the layer
                    for node in range(len(phi_v[layer])):
                        # output layer
                        if layer is NUM_LAYERS-1:
                            deltas[layer].append(
                                errors[node] * phi_prime(phi_v[layer][node]))

                        # hidden layer
                        else:
                            sum_delta_times_weights = 0
                            for next_node in range(len(phi_v[layer+1])):
                                sum_delta_times_weights += deltas[layer +
                                                                  1][next_node] * weights[layer+1][next_node][node]
                            deltas[layer].append(
                                phi_prime(phi_v[layer][node]) * sum_delta_times_weights)

                # update weights

                # deep copy to get a value copy, not reference copy
                if k is 0:
                    prev_weights = copy.deepcopy(weights)
                    prev_biases = copy.deepcopy(biases)

                for layer in range(NUM_LAYERS):
                    for from_node in range(len(weights[layer])):
                        for to_node in range(len(weights[layer][from_node])):
                            new_weight = weights[layer][from_node][to_node]

                            # for first layer, the y's are inputs
                            if layer is 0:
                                new_weight += LEARNING_RATE * \
                                    deltas[layer][from_node] * inputs[to_node]

                            # all other layers we use y's from previous layer
                            else:
                                new_weight += LEARNING_RATE * \
                                    deltas[layer][from_node] * \
                                    phi_v[layer-1][to_node]

                            # include momentum
                            new_weight += MOMENTUM * (
                                weights[layer][from_node][to_node] - prev_weights[layer][from_node][to_node])

                            # update the two weight arrays
                            prev_weights[layer][from_node][to_node] = weights[layer][from_node][to_node]
                            weights[layer][from_node][to_node] = new_weight

                        new_bias = biases[layer][from_node]

                        # adjust bias with the delta for the node
                        new_bias += LEARNING_RATE * \
                            deltas[layer][from_node]

                        # add momentum for the bias
                        new_bias += MOMENTUM * \
                            (biases[layer][from_node] -
                                prev_biases[layer][from_node])

                        # update the bias arrays
                        prev_biases[layer][from_node] = biases[layer][from_node]
                        biases[layer][from_node] = new_bias

                # END item loop

            # divide the SSE by 2 * k as per algorithm in class
            sse /= (2 * len(training_data))
            sses.append(sse)

            print("Completed epcoh {0}\tSSE = {1:.4f}".format(epoch+1, sse))

            if sses[-1] < SSE_THRESHOLD:
                print("Training complete! Time to train: {}".format(
                    datetime.datetime.now()-start))
                # ax.plot(sses)
                break

            # if epoch == NUM_EPOCHS-1:
            #     ax.plot(sses)

            # shuffle the order of the training data
            random.shuffle(training_data)

            # END epcoh loop

        classifications = []
        expected = []

        # do a forward pass to get the outputs from the network
        # loop over pieces of data in testing set
        for k in range(len(validation_data)):

            # get the specific item out of the testing set
            data = validation_data[k]
            inputs = data[:NUM_INPUT_FEATURES]

            # no longer need the data row
            del data

            testing_phi_v = []

            # loop over layers
            for layer in range(NUM_LAYERS):

                # add placeholder for next layer
                testing_phi_v.append([])

                # loop over nodes in layer
                for node in range(len(weights[layer])):

                    # variable to track induced local field
                    v = 0

                    # if the layer is the first hidden layer, use inputs instead of y's
                    # from previous layer
                    if layer is 0:
                        for i in range(len(inputs)):
                            v += weights[layer][node][i] * inputs[i]

                    else:
                        # loop over the phi_v, or y, input to the node, updating v
                        for i in range(len(testing_phi_v[layer-1])):
                            v += weights[layer][node][i] * \
                                testing_phi_v[layer-1][i]

                    # add the bias term for the node
                    v += biases[layer][node]

                    # store the phi(v) for the node
                    testing_phi_v[layer].append(phi(v))

            outputs = testing_phi_v[NUM_LAYERS-1]
            max_output = 0
            output_class = -1  # invalid to start
            idx = 0
            for output in outputs:
                if output > max_output:
                    max_output = output
                    output_class = idx
                idx += 1
            classifications.append(output_class)

        # collect the predictions
        for item in validation_data:
            expected_outputs = item[NUM_INPUT_FEATURES:]
            idx = 0
            for output in expected_outputs:
                if output == 1:
                    expected.append(idx)
                idx += 1

        cnf_matrix = confusion_matrix(expected, classifications)
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=["0", "1"],
                              title='Fold {} ({}:{}:2)'.format(fold+1, NUM_INPUT_FEATURES, LAYERS[0]))
        total_expected += expected
        total_predicted += classifications
        # END fold

    cnf_matrix = confusion_matrix(total_expected, total_predicted)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["0", "1"],
                          title='All Folds ({}:{}:2)'.format(NUM_INPUT_FEATURES, LAYERS[0]))
    plt.show()


def phi(v):
    if ACTIVATION_FUNCTION == "SIGMOID":
        return 1/(1 + math.exp(-1 * ACTIVATION_SLOPE_PARAM * v))


def phi_prime(phi_v):
    if ACTIVATION_FUNCTION == "SIGMOID":
        return ACTIVATION_SLOPE_PARAM * phi_v * (1-phi_v)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == "__main__":
    main()
