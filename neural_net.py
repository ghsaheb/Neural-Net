import os

from numpy import exp, array, random, dot, ones

from readPNG import convert_image_to_array

from scipy.special import expit

DROPOUT_PERCENT = 0.2
HIDDEN_LAYER_SIZE = 530
L2_LAMBDA = 0.01


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork:
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    @staticmethod
    def __linear(x):
        return 0.0001 * x

    @staticmethod
    def __linear_derivative(x):
        return 0.0001

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + exp(-x))
        # expit()

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def __activation_function(x, use_linear=False):
        if use_linear:
            return NeuralNetwork.__linear(x)
        return NeuralNetwork.__sigmoid(x)

    @staticmethod
    def __activation_function_derivative(x, use_linear=False):
        if use_linear:
            return NeuralNetwork.__linear_derivative(x)
        return NeuralNetwork.__sigmoid_derivative(x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, use_linear=False, use_stochastic=False, use_dropout=False, use_l2=False):
        for iteration in range(number_of_training_iterations * (training_set_inputs if use_stochastic else 1)):
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs, use_linear=use_linear, do_dropout=use_dropout, use_stochastic=use_stochastic)

            layer2_error = training_set_outputs - output_from_layer_2

            layer2_delta = layer2_error * self.__activation_function_derivative(output_from_layer_2, use_linear=use_linear)

            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)

            layer1_delta = layer1_error * self.__activation_function_derivative(output_from_layer_1, use_linear=use_linear)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta) + (L2_LAMBDA * 2 * self.layer1.synaptic_weights if use_l2 else 0)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta) + (L2_LAMBDA * 2 * self.layer2.synaptic_weights if use_l2 else 0)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    def think(self, inputs, use_linear=False, do_dropout=False, use_stochastic=False):
        if use_stochastic:
            index = int(random.rand() * len(inputs))
            inputs = array([inputs[index]])
        output_from_layer1 = self.__activation_function(dot(inputs, self.layer1.synaptic_weights), use_linear=use_linear)

        if do_dropout:
            output_from_layer1 *= random.binomial([ones(HIDDEN_LAYER_SIZE)], 1 - DROPOUT_PERCENT)[0] * (1.0 / (1 - DROPOUT_PERCENT))

        output_from_layer2 = self.__activation_function(dot(output_from_layer1, self.layer2.synaptic_weights), use_linear=use_linear)
        return output_from_layer1, output_from_layer2

    def print_weights(self):
        print("    Layer 1 (530 neurons, each with 784 inputs): ")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (10 neuron, with 530 inputs):")
        print(self.layer2.synaptic_weights)


class Dataset:
    def __init__(self, train_portion):
        self.inputs = []
        self.outputs = []
        for root, dirs, files in os.walk("notMNIST_small"):
            letter = root.split("/")[-1]
            if letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]:
                output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                output[ord(letter) - ord('A')] = 1
                count = 0
                for file in files:
                    count += 1
                    if count == 150:
                        break
                    if file == '.DS_Store':
                        continue
                    self.inputs.append(convert_image_to_array(os.path.join(root, file)))
                    self.outputs.append(output.copy())

        random.shuffle(self.inputs)
        random.shuffle(self.outputs)

        self.training_set_inputs = array(self.inputs[0:int(len(self.inputs) * train_portion)])
        self.training_set_outputs = array(self.outputs[0:int(len(self.outputs) * train_portion)])

        self.validation_set_inputs = array(self.inputs[int(len(self.inputs) * train_portion):])
        self.validation_set_outputs = array(self.outputs[int(len(self.outputs) * train_portion):])


if __name__ == "__main__":
    random.seed(1)

    # create dataset
    dataset = Dataset(1)

    # Create layer 1 (530 neurons, each with 784 inputs)
    layer1 = NeuronLayer(HIDDEN_LAYER_SIZE, 784)

    # Create layer 2 (10 neurons with each 530 inputs)
    layer2 = NeuronLayer(10, HIDDEN_LAYER_SIZE)

    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    neural_network.train(dataset.training_set_inputs, dataset.training_set_outputs, 1)

    print("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    print(neural_network.think(array([convert_image_to_array("notMNIST_small/I/R2lvdmFubmlTdGQtQmxhY2sub3Rm.png")])))