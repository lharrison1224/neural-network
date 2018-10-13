import math
import csv
import copy
import random
import datetime
import matplotlib.pyplot as plt

LEARNING_RATES = [0.01, 0.2, 0.7, 0.9]
MOMENTUM = 0.3
ACTIVATION_SLOPE_PARAM = 1
ACTIVATION_FUNCTION = "SIGMOID"
LAYERS = [11, 2]
NUM_LAYERS = len(LAYERS)
NUM_INPUT_FEATURES = 3
# the max number of epochs that will train if SSE does not converge below threshold
NUM_EPOCHS = 100
SSE_THRESHOLD = 0.001


def main():

    weights = []
    biases = []

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
    training_data = []

    # read training data
    with open("data/training.csv", "r", encoding="utf-8-sig") as csvfile:
        contents = csv.reader(csvfile)
        for row in contents:
            new_row = [float(item) for item in row]
            training_data.append(new_row)

    fig, ax = plt.subplots()
    ax.set(title="Convergence with Different Learning Rates",
           xlabel="Epoch", ylabel="SSE")

    start = datetime.datetime.now()
    for LEARNING_RATE in LEARNING_RATES:
        prev_biases = []
        prev_weights = []

        weights = []
        biases = []

        sses = []

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
                ax.plot(sses)
                break

            if epoch == NUM_EPOCHS-1:
                ax.plot(sses)

            # shuffle the order of the training data
            random.shuffle(training_data)

            # END epcoh loop

    ax.legend(LEARNING_RATES)
    plt.show()

    # begin experiement section
    # create sampling from the [-2.1, 2.1] x [-2.1, 2.1] square
    testing_data = [[round(random.uniform(-2.1, 2.1), 2),
                     round(random.uniform(-2.1, 2.1), 2), random.uniform(-0.1, 0.1)] for _ in range(1500)]

    classifications = []

    # do a forward pass to get the outputs from the network
    # loop over pieces of data in testing set
    for k in range(len(testing_data)):

        # get the specific item out of the testing set
        data = testing_data[k]
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

    # code to show scatter plot of data classifications
    # colors = ["#0000FF", "#FF0000"]
    # # we now have classifications for our testing set
    # plt.scatter([x[0] for x in testing_data], [x[1]
    #                                            for x in testing_data], c=[colors[x] for x in classifications])
    # plt.title("Cross with 3rd Feature")
    # plt.show()


def phi(v):
    if ACTIVATION_FUNCTION == "SIGMOID":
        return 1/(1 + math.exp(-1 * ACTIVATION_SLOPE_PARAM * v))


def phi_prime(phi_v):
    if ACTIVATION_FUNCTION == "SIGMOID":
        return ACTIVATION_SLOPE_PARAM * phi_v * (1-phi_v)


if __name__ == "__main__":
    main()
