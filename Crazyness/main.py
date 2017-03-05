from neuron import Neuron
from think import Think
from txtprocessor import TextProcessor
from reader import Reader
from learn import Learn

if __name__ == '__main__':
    inNeurons = [Neuron(1), Neuron(1), Neuron(1), Neuron(1), Neuron(1), Neuron(1), Neuron(1)]
    hidNeurons = [Neuron(7), Neuron(7), Neuron(7), Neuron(7), Neuron(7), Neuron(7), Neuron(7)]
    outNeurons = [Neuron(7), Neuron(7), Neuron(7)]
    thk = Think(inNeurons, hidNeurons, outNeurons)
    read = Reader()
    txtProcss = TextProcessor()
    learn = Learn(0.9, txtProcss.process(read.read()))
    learn.train(thk, inNeurons, hidNeurons, outNeurons)
