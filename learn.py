from reader import Reader
from neuron import Neuron
from txtprocessor import TextProcessor
from think import Think
import math

class Learn:
    def __init__(self, learnCoef, table):
        txtProcss = TextProcessor()
        read = Reader()
        self.learnCoef = learnCoef
        self.table = txtProcss.process(read.read())
        self.error = 0.01

    def train(self, think, inNeurons, hidNeurons, outNeurons):
        currentError = 100000.0
        while (currentError > self.error):
            stepError = 0.0
            for i in range(len(self.table)):
                outputs = think.think(self.table[i][0:7])
                think.clean()
                outErrors = [float(self.table[i][7]) - float(outputs[0]), float(self.table[i][8]) - float(outputs[1]), float(self.table[i][9]) - float(outputs[2])]
                error = self.getOutError(outErrors)
                stepError += error
                self.hidError(self.outError(outErrors, outNeurons, hidNeurons, i), outNeurons, hidNeurons, inNeurons, i)
            currentError = stepError
            print(currentError)

    def getOutError(self, outErrors):
        error = 0.0
        for i in outErrors:
            error += math.pow(i, 2)
        return error / 6

    def outError(self, outErrors, outNeurons, hidNeurons, var):
        neuronErrors = []
        for i in range(len(outNeurons)):
            error = float(self.table[var][7 + i]) * (1 - float(self.table[var][7 + i])) * outErrors[i]
            for j in range(7):
                aux_weight = outNeurons[i].getWeights()[j] + self.learnCoef * error * hidNeurons[j].getOutput()
                outNeurons[i].getWeights()[j] = aux_weight
            neuronErrors.append(error)
        return neuronErrors

    def hidError(self, neuronErrors, outNeurons, hidNeurons, inNeurons, i):
        for i in range(len(hidNeurons)):
            out = hidNeurons[i].getOutput()
            for j in range(len(inNeurons)):
                error1 = out * (1- out) * neuronErrors[0] * outNeurons[0].getWeights()[j] * inNeurons[j].getOutput()
                error2 = out * (1- out) * neuronErrors[1] * outNeurons[1].getWeights()[j] * inNeurons[j].getOutput()
                error3 = out * (1- out) * neuronErrors[2] * outNeurons[2].getWeights()[j] * inNeurons[j].getOutput()
                aux_weight = hidNeurons[i].getWeights()[i] + self.learnCoef * (error1 + error2 + error3) * inNeurons[j].getOutput()
                hidNeurons[i].getWeights()[i] = aux_weight
