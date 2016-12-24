import math, random

class Neuron:
    def __init__(self, inputs):
        self.weights = []
        self.weight_init(inputs)
        self.output = 0.0

    def pulse(self, input):
        return math.tanh(input)

    def weight_init(self, size):
        self.weights = [random.uniform(-1, 1) for i in xrange(size)]

    def setInputs(self, inputs):
        sum = 0.0
        for i in range(len(inputs)):
            sum += float(inputs[i]) * self.weights[i]
        return sum

    def getWeights(self):
        return self.weights

    def think(self, inputs):
        self.output = self.pulse(self.setInputs(inputs))
        return self.output

    def getOutput(self):
        return self.output
