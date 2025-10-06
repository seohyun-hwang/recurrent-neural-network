#include <stdio.h>
#include <math.h>


// ReLU FUNCTIONS
float defaultReLU(float input) { // ReLU function
    if (input <= 0) {
        return 0; // x < 0 => max(0, x) = 0
    }
    return input; // x > 0 => max(0, x) = x
}
float derivative1ReLU(float input) { // first derivative of ReLU function
    if (defaultReLU(input) == 0) {
        return 0; // 0' = 0
    }
    return 1; // x' = 1
}


int main(void) {
    printf("Welcome to my artificial neural network!\n\n");

    // layer/neuron indices
    int inputsetIndex = 0;
    int layerIndex = 0; // input-layer: first layer ; output-layer: last layer ; hidden-layers: the layers inbetween
    int neuronIndex = 0;
    int neuronIndex_withinOutputLayer = 0;
    // final maximum values
    int layerIndex_max = 5; // highest layerIndex
    int inputsetIndex_max = 5; // highest inputsetIndex
    int neuronIndex_maxWithinSameLayer = 5; // highest neuronIndex within same neuron-layer (input/hidden layers)
    int neuronIndex_maxWithinOutputLayer = 2; // highest neuronIndex within output-layer
    // datasets
    float weights[layerIndex_max][neuronIndex_maxWithinSameLayer][1];
    float biases[layerIndex_max][neuronIndex_maxWithinSameLayer][1];
    float layerSummationOutput[layerIndex_max - 1][1];

    float inputDatasets[inputsetIndex_max][neuronIndex_maxWithinSameLayer][0];
    float outputLayerDataPredicted[neuronIndex_maxWithinOutputLayer][1];
    float outputLayerDataDesired[1][neuronIndex_maxWithinOutputLayer][1]; // [index of the type of desired output][neuron value]



    // setting the DESIRED outputset (only)
    outputLayerDataDesired[0][0][0] = 0;
    outputLayerDataDesired[0][1][0] = 1;

    // setting the inputsets
    inputDatasets[0][0][0] = 0;
    inputDatasets[0][1][0] = 2;
    inputDatasets[0][2][0] = 5;
    inputDatasets[0][3][0] = 2;
    inputDatasets[0][4][0] = 1;

    inputDatasets[1][0][0] = 3;
    inputDatasets[1][1][0] = 0;
    inputDatasets[1][2][0] = 0;
    inputDatasets[1][3][0] = 3;
    inputDatasets[1][4][0] = 9;

    inputDatasets[2][0][0] = 4;
    inputDatasets[2][1][0] = 6;
    inputDatasets[2][2][0] = 2;
    inputDatasets[2][3][0] = 2;
    inputDatasets[2][4][0] = 0;

    inputDatasets[3][0][0] = 9;
    inputDatasets[3][1][0] = 2;
    inputDatasets[3][2][0] = 5;
    inputDatasets[3][3][0] = 2;
    inputDatasets[3][4][0] = 2;

    inputDatasets[4][0][0] = 5;
    inputDatasets[4][1][0] = 6;
    inputDatasets[4][2][0] = 5;
    inputDatasets[4][3][0] = 7;
    inputDatasets[4][4][0] = 7;

    // default-presetting all neurons' weight/bias values
    while (layerIndex < layerIndex_max) {
        while (neuronIndex < neuronIndex_maxWithinSameLayer) {
            weights[layerIndex][neuronIndex][0] = 1;
            biases[layerIndex][neuronIndex][0] = 0;

            neuronIndex++;
        }

        layerIndex++;
        neuronIndex = 0;
    }
    layerIndex = 0;

    // recurred prediction and self-correction process
    int recurrenceIndex = 1;
    while (recurrenceIndex <= 100) {

        // <forward-propagation>
        while (layerIndex < layerIndex_max) {
            while (inputsetIndex < inputsetIndex_max) {
                while (neuronIndex < neuronIndex_maxWithinSameLayer) {
                    if (layerIndex == 0) {
                        // input-layer
                        layerSummationOutput[layerIndex][0] += inputDatasets[inputsetIndex][neuronIndex][0];
                    } else if (layerIndex != layerIndex_max - 1) {
                        // hidden AND output layers
                        layerSummationOutput[layerIndex][0] += defaultReLU(weights[layerIndex][neuronIndex][0] * layerSummationOutput[layerIndex - 1][0] + biases[layerIndex][neuronIndex][0] * layerSummationOutput[layerIndex - 1][0]);
                    }
                    neuronIndex++;
                }
                if (layerIndex == layerIndex_max - 1) {
                    // output layer (apply ReLU, then Softmax)
                    while (neuronIndex_withinOutputLayer < neuronIndex_maxWithinOutputLayer) {
                        outputLayerDataPredicted[neuronIndex_withinOutputLayer][0] = defaultReLU(weights[layerIndex][neuronIndex][0] * layerSummationOutput[layerIndex - 1][0] + biases[layerIndex][neuronIndex][0] * layerSummationOutput[layerIndex - 1][0]);
                        neuronIndex_withinOutputLayer++;
                    }
                    neuronIndex_withinOutputLayer = 0;
                    float eZjSum = 0;
                    while (neuronIndex_withinOutputLayer < neuronIndex_maxWithinOutputLayer) {
                        // make a sum of e^zj for first stage of Softmax
                        float eZj = outputLayerDataPredicted[neuronIndex_withinOutputLayer][0];
                        eZjSum += expf(eZj);
                        neuronIndex_withinOutputLayer++;
                    }
                    neuronIndex_withinOutputLayer = 0;
                    while (neuronIndex_withinOutputLayer < neuronIndex_maxWithinOutputLayer) {
                        // complete the Softmax
                        float eZi = expf(layerSummationOutput[layerIndex][0]);
                        outputLayerDataPredicted[neuronIndex_withinOutputLayer][0] = eZi / eZjSum;
                        neuronIndex_withinOutputLayer++;
                    }
                }
                neuronIndex = 0;
                inputsetIndex++;
            }
            inputsetIndex = 0;
            layerIndex++;
        }
        for (int i = 0; i < layerIndex_max; i++) {
            layerSummationOutput[i][0] = 0;
        }
        // </forward-propagation>


        // <cross-entropy-loss-check>
        float crossEntropyLoss = 0;
        neuronIndex_withinOutputLayer = 0;
        while (neuronIndex_withinOutputLayer < neuronIndex_maxWithinOutputLayer) {
            crossEntropyLoss += outputLayerDataDesired[0][neuronIndex_withinOutputLayer][0] * logf(outputLayerDataPredicted[neuronIndex_withinOutputLayer][0]);
            neuronIndex_withinOutputLayer++;
        }
        crossEntropyLoss *= -1;
        printf("\nCross-Entropy-Loss for recurrence-iteration %d: %f", recurrenceIndex, crossEntropyLoss);
        neuronIndex_withinOutputLayer = 0;
        // </cross-entropy-loss-check>


        // <backpropagation> // error-contribution tracing of biases and weights

        float dC_dYhat = 0; // calculating the last term of d(C)/d(param) [the value of the contribution of the predicted output value to the overall cost]
        neuronIndex_withinOutputLayer = 0;
        while (neuronIndex_withinOutputLayer < neuronIndex_maxWithinOutputLayer) {
            dC_dYhat += (outputLayerDataDesired[0][neuronIndex_withinOutputLayer][0] / outputLayerDataPredicted[neuronIndex_withinOutputLayer][0]);
            neuronIndex_withinOutputLayer++;
        }
        dC_dYhat *= -1;
        neuronIndex_withinOutputLayer = 0;

        // modification of weights/biases through backpropagation
        layerIndex = 1;
        while (layerIndex < layerIndex_max) {
            weights[layerIndex][neuronIndex][0] -= layerSummationOutput[layerIndex - 1][0] * derivative1ReLU(biases[layerIndex][neuronIndex][0] + weights[layerIndex][neuronIndex][0] * layerSummationOutput[layerIndex - 1][0]) * dC_dYhat;
            biases[layerIndex][neuronIndex][0] -= derivative1ReLU(biases[layerIndex][neuronIndex][0] + weights[layerIndex][neuronIndex][0] * layerSummationOutput[layerIndex - 1][0]) * dC_dYhat;
            layerIndex++;
        }
        layerIndex = 0;

        // </backpropagation>

        recurrenceIndex++;
        layerIndex = 0;
    }

    return 0;
}





// previous draft
/*

// SUPPORT-VARIABLES
//for datasets
int countTotalIndependent_inputsets_trainingSet = 100; // the count of inputsets in the training-dataset => the count of how many predicted-outputsets should be derived
int count_dimensionsPerInputset_general = 256; // 16px x 16px greyscale image (working with 1 color)

int countTotalIndependent_desiredOutputsets_general = 10; // 0,1,2,3,4,5,6,7,8,9
int count_elementsPerOutputset_general = 10; // 0,1,2,3,4,5,6,7,8,9

//for neurons
int countTotal_neurons_general = 50;
int countTotal_layers_general = 5;
int neuronLayer_firstElementPointers[countTotal_layers_general]; // pointer to the first element of each separate unique neuron-layer regarding a weight/bias/output array


// ANN DATASETS

//desired-outputset definition
float desiredOutputsets[count_elementsPerOutputset_general * countTotalIndependent_desiredOutputsets_general]; // elements of desired-outputsets placed mext to each other
int desiredOutputsets_firstElementPointers[countTotalIndependent_desiredOutputsets_general]; // pointer to the first element of each separate unique desired-outputset in above array

//predicted-outputset definitions
float predictedOutputsets[count_elementsPerOutputset_general * countTotalIndependent_inputsets_trainingSet]; // elements of predicted-outputsets placed mext to each other
int predictedOutputsets_firstElementPointers[countTotalIndependent_inputsets_trainingSet]; // pointer to the first element of each separate unique predicted-outputset in above array

//cross-entropy-loss
float crossEntropyLoss[countTotalIndependent_inputsets_trainingSet];


// ANN INNER-ARCHITECTURE (hidden layers, output layer)

int neuronCount_perLayer[countTotal_layers_general];

// [layerIndex] [neuronIndex] [inputsetIndex] [inputDimensionIndex] [weight-value]
float weights[countTotal_layers_general][countTotal_neurons_general][countTotalIndependent_inputsets_trainingSet][count_dimensionsPerInputset_general][1];

// [layerIndex] [neuronIndex] [inputsetIndex] [bias-value]
float biases[countTotal_layers_general][countTotal_neurons_general][countTotalIndependent_inputsets_trainingSet][1];

// [layerIndex] [neuronIndex] [output-value]
float outputs[countTotal_layers_general][countTotal_neurons_general][1];


// MAIN FUNCTION
int main(void) {
    printf("Welcome to my artificial neural network!");

    // default-presetting all neurons' weight/bias values
    for (int i = 0; i < countTotal_layers_general; i++) {
        for (int j = 0; j < neuronCount_perLayer[i]; j++) {
            for (int layerIndex = 0; layerIndex < countTotal_layers_general; layerIndex++) {
                for (int neuronIndex = 0; neuronIndex < countTotal_neurons_general; neuronIndex++) {
                    for (int inputsetIndex = 0; inputsetIndex < countTotalIndependent_inputsets_trainingSet; inputsetIndex++) {
                        for (int inputDimensionIndex = 0; inputDimensionIndex < count_dimensionsPerInputset_general; inputDimensionIndex++) {
                            weights[layerIndex][neuronIndex][inputsetIndex][inputDimensionIndex][0] = 1;
                        }
                        biases[layerIndex][neuronIndex][inputsetIndex][0] = 0;
                    }
                }
            }
        }
    }


    // recurrence begins here
    int recurrenceCount = 0;
    while (recurrenceCount < 100) { // 100 iterations

        // looping until training set is exhausted
        for (int m = 0; m < countTotalIndependent_inputsets_trainingSet / countTotalIndependent_desiredOutputsets_general; m++) {

            for (int k = 0; k < countTotalIndependent_desiredOutputsets_general; k++) {

                crossEntropyLoss(k) = 0; // cross-entropy loss value


                //forward pass
                for (int i = 0; i < countTotal_layers_general; i++) {
                    for (int j = 0; j < neuronCount_perLayer[i]; j++) {
                        for (int layerIndex = 1; layerIndex < countTotal_layers_general; layerIndex++) {

                            for (int neuronIndex = 0; neuronIndex < countTotal_neurons_general; neuronIndex++) {

                                for (int inputsetIndex = 0; inputsetIndex < countTotalIndependent_inputsets_trainingSet; inputsetIndex++) {
                                    for (int inputDimensionIndex = 0; inputDimensionIndex < count_dimensionsPerInputset_general; inputDimensionIndex++) {

                                        outputs[layerIndex][neuronIndex][0] += (weights[layerIndex][neuronIndex][inputsetIndex][inputDimensionIndex][0] * outputs[layerIndex - 1][neuronIndex][0]); // contributing to the sum of the previous layer's neurons

                                    }
                                    outputs[layerIndex][neuronIndex][0] += biases[layerIndex][neuronIndex][inputsetIndex][0];
                                }
                                // ReLU activation
                                outputs[layerIndex][neuronIndex][0] = defaultReLU(outputs[layerIndex][neuronIndex][0]);
                            }

                            if (layerIndex == countTotal_layers_general - 1) { // last layer
                                //unload output values, apply Softmax on them, then load those Softmax-modified values into predictedOutputs
                                float zjSum = 0; // make a zj-sum first (for Softmax)
                                for (int l = predictedOutputsets_firstElementPointers(k - 1); l < predictedOutputsets_firstElementPointers(k); l++) { // traverse through predictedOutputsets to gauge how many outputs are being dealt with
                                    zjSum += expf(output[i]);
                                }
                                for (int l = predictedOutputsets_firstElementPointers(k - 1); l < predictedOutputsets_firstElementPointers(k); l++) {
                                    predictedOutputsets(l) = expf(output(predictedOutputsets(k) + l)) / zjSum; // apply zj-sum to each output of the output layer
                                }

                                // find the cross-entropy-loss
                                float celTemp = 0;
                                for (int l = predictedOutputsets_firstElementPointers(k - 1); l < predictedOutputsets_firstElementPointers(k); l++) { // traverse through predictedOutputsets to gauge how many outputs are being dealt with
                                    celTemp += desiredOutputsets[j] * logf(output[i]);
                                }
                                celTemp *= -1;
                                crossEntropyLoss(k) += celTemp;
                            }
                        }
                    }
                }
            }
            // error-contribution tracing (weight and bias simultaneously for each neuron) with nonstochastic gradient-descent implementation
            float dC_dYhat = 0; // calculating the last term of d(C)/d(param)
            for (int i = 0; i < countTotalIndependent_inputsets_trainingSet; i++) {
                dC_dYhat += crossEntropyLoss[i] / output[i];
            }
            dC_dYhat *= -1;

            // modification of weights/biases through backpropagation
            for (int i = countTotal_layers_general; i > 0; i--) { // going layer by layer
                //calculating the sum of the previous layer's neurons
                float sumOfNeuronOutputs_previousLayer = 0;
                for (int j = neuronLayer_firstElementPointers(i - 1); j < neuronLayer_firstElementPointers(i); j++) {
                    sumOfNeuronOutputs_previousLayer += output(j); // contributing to the sum of the previous layer's neurons
                }
                weight[i] -= sumOfNeuronOutputs_previousLayer * derivative1ReLU(bias[i] + weight[i] * sumOfNeuronOutputs_previousLayer) * dC_dYhat;
                bias[i] -= derivative1ReLU(bias[i] + weight[i] * sumOfNeuronOutputs_previousLayer) * dC_dYhat;
            }
        }

        recurrenceCount++;
    }





    // evaluation of model on testing-set (coming later...)
}
*/