# -*- coding: utf-8 -*-
# Imports.
import numpy as np
import math
# Classes for a basic neural network.

# Class for a single neuron. Specialized to use with a multi-layer feed-
# forward neural network using the logistic sigmoid activation funcion.
class Neuron:
    # Fields:
    #   _aWeights - Array of the neuron's weights.
    #   _fcnActFcn - Activation function used to compute the output.
    #   _dBeta - Scale factor applied to the activation function.
    #   _dActLevel - Actvity level of the neuron.
    
    # Constructor.
    def __init__(self, iNumInputs, fcnAF, dBeta = 1.0):
        # Initialize the weights to random values. Distribution is uniform in
        # range -1/sqrt(iNumInputs) to 1/sqrt(iNumInputs).
        dSqrtNumInputs = 1/math.sqrt(iNumInputs)
        self._aWeights = np.random.uniform(-dSqrtNumInputs, dSqrtNumInputs, iNumInputs)
        # The activation function and scale factor are passed in.
        self._fcnActFcn = fcnAF
        self._dBeta = dBeta
        # Initialize the activity level to zero.
        self._dActLevel = 0.0
    
    # Calculate the output, given a vector of inputs.
    def dOutput(self, aInputs):
        # Calculate the activity level.
        self._dActLevel = np.dot(self._aWeights, aInputs)
        # Calculate and return the output.
        dOut = self._fcnActFcn(self._dBeta * self._dActLevel)
        return dOut
    
    # Adjust the weights to reduce the error in the output for the given input.
    def vLearn(self, aInputs, dError, dEta):
        # Calculate the output from this neuron.
        dY = self.dOutput(aInputs)
        # Calculate the factor to multiply aInputs by when adjusting weights.
        dFactor = dEta * dError * self._dBeta * dY * (1.0 - dY)
        # Adjust the weights by adding on the factor times the inputs.
        self._aWeights += dFactor * aInputs

# Possible activation functions.
# Heaviside fcn.
def Heaviside(dX):
    if dX < 0.0:
        return 0.0
    else:
        return 1.0

# Logistic sigmoid function.
def Sigmoid(dX):
    return 1.0 / (1.0 + math.exp(-dX))

# A class for a group of neurons arranged into a single layer. Any input put into
# the layer is fed into each neuron in the layer, and the outputs from all of the
# neurons are combined into a vector, which becomes the output of the layer.
class NNLayer:
    # Field:
    #   _aNeurons - Array of the neurons in this layer.
    
    # Constructor.
    # Parameters are the same as for the Neuron class, except that we also need
    # a parameter for the number of neurons in the layer.
    def __init__(self, iNumInputs, iNumNeurons, fActFcn, dBeta = 1.0):
        # Bulid up a list of the proper number of neurons, then convert the list
        # into an array.
        lstNeurons = []
        for iCount in range(iNumNeurons):
            lstNeurons.append(Neuron(iNumInputs, fActFcn, dBeta))
        # Create the array and put it into the field _aNeurons.
        self._aNeurons = np.array(lstNeurons)
        # For convenience, keep track of the the number of inputs and number of
        # neurons.
        self._iNumInputs = iNumInputs
        self._iNumNeurons = iNumNeurons
    
    # Calculate the array of outputs from all neurons in the layer, given an
    # array of inputs.
    def aOutput(self, aInputs):
        # Create a list of the outputs from each of the neurons in the layer,
        # then convert it to an array.
        lstOuts = []
        for neuOne in self._aNeurons:
            lstOuts.append(neuOne.dOutput(aInputs))
        # Finished. Counvert list to an array and return it.
        return np.array(lstOuts)
    
    # Adjust the weights of all of the neurons in this layer, given an input
    # array and an array of errors in the outputs of the neurons. At the same
    # time, calculate the errors in the outputs of the previous layer and
    # return them.
    def aLearn(self, aInputs, aErrors, dEta):
        # Loop through the neurons in this layer and calculate each one's
        # contribution to the errors in the previous layer, then adjust its
        # weights. When finished, return the array of errors in the previous
        # layer.
        # Start with zeros for the errors in the previous layer.
        aPrevErrors = np.zeros((self._iNumInputs,))
        for k in range(self._iNumNeurons):
            # Get the current (kth) neuron, calcualte its output, and get its
            # error.
            neuCurrNeu = self._aNeurons[k]
            dY = neuCurrNeu.dOutput(aInputs)
            dCurrError = aErrors[k]
            dFactor = dY * (1.0 - dY) * dCurrError
            # This neuron's contribution to the error in the previous layer is
            # the above factor times its weights.
            aPrevErrors += dFactor * neuCurrNeu._aWeights
            # Now we can adjust this neuron's weights.
            neuCurrNeu.vLearn(aInputs, dCurrError, dEta)
        
        # Once we have finished the loop, all neurons have had their weights
        # updated, and we have calculated the contribution of all neurons to
        # the errors in the previous layer. Return the array of errors in the
        # previous layer.
        return aPrevErrors

# A class for a full multi-layer neural network.
class NeuralNet:
    # Fields:
    #   _aLayers - Array of the layers in the network.
    #   _iNumLayers - Number of layers.
    #   _iNumInputs - Number of inputs to the first layer.
    #   _iNumOutputs - Number of outputs from the final layer.
    
    # Constructor.
    def __init__(self, iNumInputs, lstNumNeurons, fActFcn, dBeta = 1.0):
        # Bulid up the list of layers one by one.
        lstLayers = []
        # Loop through list of numbers of neurons in each layer, create each
        # layer in turn. Have to keep track of the number of inputs for the
        # current layer, because it changes from layer to layer.
        iCurrNumInputs = iNumInputs
        for iCurrNumNeurons in lstNumNeurons:
            lstLayers.append(NNLayer(iCurrNumInputs, iCurrNumNeurons, fActFcn, dBeta))
            # Number of inputs for the next layer is the number of neurons in
            # the current layer.
            iCurrNumInputs = iCurrNumNeurons
        # List of layers complete. Convert to an array and put into field
        # _aLayers. Also, set other fields.
        self._aLayers = np.array(lstLayers)
        self._iNumLayers = len(self._aLayers)
        self._iNumInputs = iNumInputs
        self._iNumOutputs = lstNumNeurons[-1]
    
    # Compute the output, given an array of inputs.
    def aOutput(self, aInputs):
        # Loop through the layers in the network, putting the appropriate
        # input into each layer and having it calculate the corresponding
        # output. The output from one layer becomes the input to the next
        # layer. The output from the last layer is the final output.
        aCurrInputs = aInputs
        for nnlOneLayer in self._aLayers:
            aCurrOutputs = nnlOneLayer.aOutput(aCurrInputs)
            aCurrInputs = aCurrOutputs
        # Finished. Outputs from last time through loop are the outputs from
        # the network.
        return aCurrOutputs
    
    # Adjust the weights in all of the neurons in all of the layers of this
    # network, given the initial input and an array of the targets.
    def vLearn(self, aInputs, aTargets, dEta):
        # Start by building up a list of the layers together with their inputs.
        aCurrInputs = aInputs
        lstLayersInputs = []
        for nnlCurrLayer in self._aLayers:
            # Add tuple of current layer and its inputs to the list.
            lstLayersInputs.append((aCurrInputs, nnlCurrLayer))
            # Update the current inputs: The inputs to the next layer are the
            # outputs from the current layer.
            aCurrOutputs = nnlCurrLayer.aOutput(aCurrInputs)
            aCurrInputs = aCurrOutputs
        # Now we have a list of the layers and inputs. Now, calculate errors
        # and adjust the weights in each layer, working from the last layer
        # back to the first layer.
        # Note that "aCurrOutputs" at the end of the for loop is the output
        # from the last layer. We use it
        # together with the array of targets to calculate the errors in the
        # last layer.
        aCurrErrors = aTargets - aCurrOutputs
        # Loop through the the layers from the last to the first, training each
        # layer and getting the errors for the previous layer.
        for tLayerInputs in lstLayersInputs[::-1]:
            # Extract current layer and inputs from the tuple.
            (aCurrInputs, nnlCurrLayer) = tLayerInputs
            # Update weights in the current layer, obtaining at the same time
            # the errors in the previous layer, which will be the next layer
            # we train.
            aCurrErrors = nnlCurrLayer.aLearn(aCurrInputs, aCurrErrors, dEta)
    
    # Train the neural network on a given training set, going through the set
    # single time.
    def vLearnOnePass(self, tsSet, dEta):
        # Loop through the training pairs in the training set. Train on each
        # one individually.
        for tOnePair in tsSet:
            # Extract the inputs and targets.
            (aInputs, aTargets) = tOnePair
            # Train this neural network with the given inputs and targets.
            self.vLearn(aInputs, aTargets, dEta)
    
    # Train the network on a given training set by going through the training
    # set multiple times.
    def vLearnManyPasses(self, tsSet, dEta, iNumReps):
        # Repeat calling vLearnOnePass multiple times.
        for k in range(iNumReps):
            # Make a copy of the training set, then shuffle it in a random order,
            # and train using the shuffled set.
            tsShuffled = list(tsSet)
            np.random.shuffle(tsShuffled)
            # Train ourselves using the shuffled training set.
            self.vLearnOnePass(tsShuffled, dEta)
    
    # Find the maximum difference between an output and the corresponding target
    # for all of the examples (pairs) in a given training set.
    def dShowMaxError(self, tsSet):
        # Keep track of the largest differnce found so far. Initially 0.
        dMaxDiffSoFar = 0.0
        # Loop through the training set and compare outputs with targets.
        for (aInputs, aTarget) in tsSet:
            # Calculate output, then difference between output and target.
            aOutput = self.aOutput(aInputs)
            dDiff = np.linalg.norm(aOutput - aTarget)
            # If new difference is bigger than max difference so far, update
            # max difference so far.
            if dDiff > dMaxDiffSoFar:
                dMaxDiffSoFar = dDiff
        # Finished with loop, so max diff so far is the overall max difference.
        return dMaxDiffSoFar
    
    # Show what fraction of the training pairs yield outputs that are within
    # a given tolerance of their targets.
    def dShowTSPerform(self, tsSet, dTolerance):
        # Set up to count the number of training pairs for which the output is
        # within the given tolerance of the target.
        iCount = 0
        for (aInputs, aTarget) in tsSet:
            # Calculate output, then difference between output and target.
            aOutput = self.aOutput(aInputs)
            dDiff = np.linalg.norm(aOutput - aTarget)
            # Check whether the difference is less than or equal to the tolerance.
            # If so, increment the count.
            if dDiff <= dTolerance:
                iCount += 1
        # Checked all pairs. Calculate fraction and return it.
        return iCount / len(tsSet)
