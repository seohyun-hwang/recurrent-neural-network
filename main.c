#include <stdio.h>
#include <math.h>






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

// [layerIndex] [neuronIndex] [inputsetIndex] [inputDimensionIndex] [output]
float weights[countTotal_layers_general][countTotal_neurons_general][countTotalIndependent_inputsets_trainingSet][count_dimensionsPerInputset_general][1];

// [layerIndex] [neuronIndex] [inputsetIndex] [output]
float biases[countTotal_layers_general][countTotal_neurons_general][countTotalIndependent_inputsets_trainingSet][1];


// SUPPORT FUNCTIONS
float defaultReLU(float input) { // ReLU function
    if (input <= 0) {
        return 0; // x < 0 --> max(0, x) = 0
    }
    return input; // x > 0 --> max(0, x) = x
}
float derivative1ReLU(float input) { // first derivative of ReLU function
    if (defaultReLU(input) != 0) {
        return 1; // x' = 1
    }
    return 0; // 0' = 0
}


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

                            //calculating the sum of the previous layer's neurons
                            float sumOfNeuronOutputs_previousLayer = 0;
                            for (int neuronIndex = 0; neuronIndex < countTotal_neurons_general; neuronIndex++) {

                                for (int inputsetIndex = 0; inputsetIndex < countTotalIndependent_inputsets_trainingSet; inputsetIndex++) {
                                    for (int inputDimensionIndex = 0; inputDimensionIndex < count_dimensionsPerInputset_general; inputDimensionIndex++) {
                                        weights[layerIndex][neuronIndex][inputsetIndex][inputDimensionIndex][0] = ;

                                    }
                                    biases[layerIndex][neuronIndex][inputsetIndex][0] = ;
                                }
                            }
                        }
                    }
                }
            }
        }

        recurrenceCount++;
    }





    // recurrence begins here
    //int recurrenceCount = 0;
    while (recurrenceCount < 100) { // 100 iterations

        // looping until training set is exhausted
        for (int m = 0; m < countTotalIndependent_inputsets_trainingSet / countTotalIndependent_desiredOutputsets_general; m++) {

            for (int k = 0; k < countTotalIndependent_desiredOutputsets_general; k++) {

                crossEntropyLoss(k) = 0; // cross-entropy loss value

                // forward-pass
                for (int i = 0; i < countTotal_layers_general; i++) { // going layer by layer
                    //calculating the sum of the previous layer's neurons
                    float sumOfNeuronOutputs_previousLayer = 0;
                    for (int j = neuronLayer_firstElementPointers(i - 1); j < neuronLayer_firstElementPointers(i); j++) {
                        sumOfNeuronOutputs_previousLayer += output(j); // contributing to the sum of the previous layer's neurons
                    }

                    //deriving the output of each neuron of the current layer (using the sum derived from the loop above)
                    for (int j = neuronLayer_firstElementPointers(i); j < neuronLayer_firstElementPointers(i + 1); j++) { // going neuron by neuron within each layer
                        output(j) = defaultReLU( bias(j) + (weight(j) * sumOfNeuronOutputs_previousLayer) );
                    }

                    //unload output values, apply Softmax on them, then load those Softmax-modified values into predictedOutputs
                    float zjSum = 0; // make a zj-sum first (for Softmax)
                    for (int j = predictedOutputsets_firstElementPointers(k - 1); j < predictedOutputsets_firstElementPointers(k); j++) { // traverse through predictedOutputsets to gauge how many outputs are being dealt with
                        zjSum += expf(output[i]);
                    }
                    for (int j = predictedOutputsets_firstElementPointers(k - 1); j < predictedOutputsets_firstElementPointers(k); j++) {
                        predictedOutputsets(j) = expf(output(predictedOutputsets(k) + j)) / zjSum; // apply zj-sum to each output of the output layer
                    }

                    // find the cross-entropy-loss
                    float celTemp = 0;
                    for (int j = predictedOutputsets_firstElementPointers(k - 1); j < predictedOutputsets_firstElementPointers(k); j++) { // traverse through predictedOutputsets to gauge how many outputs are being dealt with
                        celTemp += desiredOutputsets[j] * logf(output[i]);
                    }
                    celTemp *= -1;
                    crossEntropyLoss(k) += celTemp;
                }
                crossEntropyLoss(k) /= float (countTotalIndependent_desiredOutputsets_general);
                printf("Overall Cross-Entropy Loss for iteration %d = %f", recurrenceCount, crossEntropyLoss(k));
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

        recurrenceCount++;
    }


    // evaluation on testing set (coming later... stratification is required)
}
