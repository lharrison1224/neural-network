import math
import csv

LEARNING_RATE = 0.7
MOMENTUM = 0.3
ACTIVATION_SLOPE_PARAM = 1
ACTIVATION_FUNCTION = "SIGMOID"


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

    # initialize empty training data array
    training_data = []

    # read training data
    with open("data/training.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            new_row = [float(item) for item in row]
            training_data.append(new_row)

    # loop for number of epochs
    for epoch in range(1):

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
            for layer in range(2):

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
            print(phi_v)
            # END layer loop

        # END item loop

    # END epcoh loop


def phi(v):
    if ACTIVATION_FUNCTION is "SIGMOID":
        return 1/(1 + math.exp(-1 * ACTIVATION_SLOPE_PARAM * v))


def phi_prime(v):
    if ACTIVATION_FUNCTION is "SIGMOID":
        return ACTIVATION_SLOPE_PARAM * (phi(v)) * (1-phi(v))


if __name__ == "__main__":
    main()
