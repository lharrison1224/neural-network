import math
import csv
import copy

LEARNING_RATE = 0.7
MOMENTUM = 0.3
ACTIVATION_SLOPE_PARAM = 1
ACTIVATION_FUNCTION = "SIGMOID"
NUM_LAYERS = 2
NUM_EPOCHS = 1


def main():

    # initialize empty weights arrays
    weights_1 = []
    weights_2 = []

    # initialize empty bias arrays
    biases_1 = []
    biases_2 = []

    # read biases for first layer
    with open("data/b1.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            biases_1.append(float(row[0]))

    # read the weights for first layer
    with open("data/w1.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            new_row = [float(item) for item in row]
            weights_1.append(new_row)

    # read the biases for the second layer
    with open("data/b2.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            biases_2.append(float(row[0]))

    # read the weights for the second layer
    with open("data/w2.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            new_row = [float(item) for item in row]
            weights_2.append(new_row)

    weights = [weights_1, weights_2]
    biases = [biases_1, biases_2]
    prev_weights = []
    prev_biases = []

    # initialize empty training data array
    training_data = []

    # read training data
    with open("data/training.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            new_row = [float(item) for item in row]
            training_data.append(new_row)

    # loop for number of epochs
    for epoch in range(NUM_EPOCHS):

        # loop over pieces of data in training set
        for k in range(len(training_data)):

            # get the specific item out of the training set
            data = training_data[k]

            # split the item into its inputs and outputs
            inputs = data[:3]
            expected_outputs = data[3:]

            # no longer need the data row
            del data

            # set the initial phi_v to the inputs for convenience
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
                            v += weights[layer][node][i] * phi_v[layer-1][i]

                    # add the bias term for the node
                    v += biases[layer][node]

                    # store the phi(v) for the node
                    phi_v[layer].append(phi(v))

                    # END node loop

                # END layer loop

            # loop over output nodes
            errors = []
            for output_node in range(len(phi_v[NUM_LAYERS-1])):
                errors.append(
                    expected_outputs[output_node] - phi_v[NUM_LAYERS-1][output_node])

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

        print("Completed epcoh {}".format(epoch))
        # END epcoh loop


def phi(v):
    if ACTIVATION_FUNCTION is "SIGMOID":
        return 1/(1 + math.exp(-1 * ACTIVATION_SLOPE_PARAM * v))


def phi_prime(phi_v):
    if ACTIVATION_FUNCTION is "SIGMOID":
        return ACTIVATION_SLOPE_PARAM * phi_v * (1-phi_v)


if __name__ == "__main__":
    main()
