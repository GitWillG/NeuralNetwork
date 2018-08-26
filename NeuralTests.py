# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:14:32 2018

@author: wgrant14
"""
import numpy as np

import NeuralNet as nn

aPimaRawData = np.loadtxt('pima.data', delimiter = ',')

aAllInputs = aPimaRawData[:, :8]
aAllTargets = aPimaRawData[:, 8:]

aMins = aAllInputs.min(axis = 0)
aMaxs = aAllInputs.max(axis = 0)
aSpreads = aMaxs - aMins
aAllInputsNorm = (aAllInputs - aMins) / aSpreads


tsPima = [(aAllInputsNorm[k], aAllTargets[k]) for k in range(aAllInputs.shape[0])]
# Split into actual training set, validation set, and test set.
tsPimaTrain = tsPima[0::2]
tsPimaValid = tsPima[1::4]
tsPimaTest = tsPima[3::4]




np.random.seed(10)


#nnPima = nn.NeuralNet(8, [28, 6, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
 
#nnPima = nn.NeuralNet(8, [12, 6, 1], nn.Sigmoid, 3.0)
#iBlockSize = 200
#Finished
#        Number of repetitions: 1800
#        Final fraction correct: 0.4739583333333333


#nnPima = nn.NeuralNet(8, [12, 6, 1], nn.Sigmoid, 1.0)
#iBlockSize = 200
#Finished
#        Number of repetitions: 400
#        Final fraction correct: 0.25


#nnPima = nn.NeuralNet(8, [8, 6, 1], nn.Sigmoid, 3.0)
#iBlockSize = 200
#Finished
#        Number of repetitions: 600
#        Final fraction correct: 0.3541666666666667


#nnPima = nn.NeuralNet(8, [8, 6, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 1500
#        Final fraction correct: 0.3958333333333333


#nnPima = nn.NeuralNet(8, [16, 6, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 2500
#        Final fraction correct: 0.5833333333333334
                    

#nnPima = nn.NeuralNet(8, [16, 8, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 3500
#        Final fraction correct: 0.5625


#nnPima = nn.NeuralNet(8, [16, 4, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 3500
#        Final fraction correct: 0.53125


#nnPima = nn.NeuralNet(8, [16, 8, 3], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 2000
#        Final fraction correct: 0.4947916666666667



#nnPima = nn.NeuralNet(8, [16, 7, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 2000
#        Final fraction correct: 0.484375


#nnPima = nn.NeuralNet(8, [16, 5, 1], nn.Sigmoid, 3.0)
#iBlockSize = 500
#Finished
#        Number of repetitions: 3500
#        Final fraction correct: 0.5833333333333334


nnPima = nn.NeuralNet(8, [32, 6, 1], nn.Sigmoid, 3.0)
iBlockSize = 500
#Finished
#        Number of repetitions: 3500
#        Final fraction correct: 0.6041666666666666


                   


# Repeat training in blocks of 100 repetitions. Continue to train as long as
# the fraction of outputs within 0.1 of their targets increases.

iBlockCount = 0
iRepetitions = 0
dCurrFracCorrect = 0.0
bContinue = True
while bContinue:
    iBlockCount += 1
    print('Block number', iBlockCount, flush=True)
    nnPima.vLearnManyPasses(tsPimaTrain, 0.25, iBlockSize)
    iRepetitions += iBlockSize
    dOldFracCorrect = dCurrFracCorrect
    dCurrFracCorrect = nnPima.dShowTSPerform(tsPimaValid, 0.1)
    print('Fraction correct', dCurrFracCorrect, flush=True)
    bContinue = (dCurrFracCorrect > dOldFracCorrect)
# Finished. Print final results.
print('Finished')
print('\tNumber of repetitions:', iRepetitions)
print('\tFinal fraction correct:', nnPima.dShowTSPerform(tsPimaTest, 0.1))






