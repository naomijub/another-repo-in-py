from neuron import Neuron

class Think:
    def __init__(self, inNeuron, hidNeuron, outNeuron):
        self.inNeuron = inNeuron
        self.hidNeuron = hidNeuron
        self.outNeuron = outNeuron
        self.intermediateInState = []
        self.intermediateOutState = []
        self.out = []

    def think(self, inputs):
        for i in range(len(inputs)):
            self.intermediateInState.append(self.inNeuron[i].think([inputs[i]]))

        for i in range(len(self.hidNeuron)):
            self.intermediateOutState.append(self.hidNeuron[i].think(self.intermediateInState))

        for i in range(len(self.outNeuron)):
            self.out.append(self.outNeuron[i].think(self.intermediateOutState))

        return self.out

    def clean(self):
        self.intermediateInState = []
        self.intermediateOutState = []
        self.out = []
