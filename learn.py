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
        currenterror = 100000.0
        while (currentError > self.error):
            steperror = 0.0
            for i in range(len(self.table)):
                outputs = think.think(self.table[i][0:7])
                think.clean()
                outerrors = [float(self.table[i][7]) - float(outputs[0]), float(self.table[i][8]) - float(outputs[1]), float(self.table[i][9]) - float(outputs[2])]
                error = self.get_out_error(outErrors)
                steperror += error
                self.hidError(self.outError(outErrors, outNeurons, hidNeurons, i), outNeurons, hidNeurons, inNeurons, i)
            currenterror = steperror
            print(currenterror)

    def get_out_error(self, outerrors):
        error = 0.0
        for i in outerrors:
            error += math.pow(i, 2)
        return error / 6

    def out_error(self, outErrors, outNeurons, hidNeurons, var):
        neuronErrors = []
        for i in range(len(outNeurons)):
            error = float(self.table[var][7 + i]) * (1 - float(self.table[var][7 + i])) * outErrors[i]
            for j in range(7):
                aux_weight = outNeurons[i].get_weights()[j] + self.learnCoef * error * hidNeurons[j].getOutput()
                outNeurons[i].getW_wights()[j] = aux_weight
            neuronErrors.append(error)
        return neuronErrors

    def hidError(self, neuronErrors, outNeurons, hidNeurons, inNeurons, i):
        for i in range(len(hidNeurons)):
            out = hidNeurons[i].getOutput()
            for j in range(len(inNeurons)):
                error1 = out * (1- out) * neuronErrors[0] * outNeurons[0].get_weights()[j] * inNeurons[j].get_output()
                error2 = out * (1- out) * neuronErrors[1] * outNeurons[1].get_weights()[j] * inNeurons[j].get_output()
                error3 = out * (1- out) * neuronErrors[2] * outNeurons[2].get_weights()[j] * inNeurons[j].get_output()
                aux_weight = hidNeurons[i].get_weights()[i] + self.learnCoef * (error1 + error2 + error3) * inNeurons[j].get_output()
                hidNeurons[i].get_weights()[i] = aux_weight
