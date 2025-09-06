#include <stdio.h>
#include <math.h>

// ReLU function
float defaultReLU(float input) {
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

int main(void) {
    printf("Welcome to my artificial neural network!");

    int total_recurrenceCount = 100; // how many times this recurrent neural network should actually do the recurring before calling it a day


    int total_neuronCount;
    int total_layerCount;
    int total_uniqueDesiredOutputsetCount;

    int layer_firstElementPointer[total_layerCount]; // all layers incl. the input layer and incl. the position immediately after total_neuronCount
    int outputLayer_neuronCount = total_neuronCount - layer_firstElementPointer(total_layerCount - 1);

    float weight[total_neuronCount];
    float bias[total_neuronCount];
    float output[total_neuronCount];
    float weightContributionToError[total_neuronCount];
    float biasContributionToError[total_neuronCount];

    float desiredOutput[outputLayer_neuronCount * total_uniqueDesiredOutputsetCount];
    int desiredOutput_firstElementPointer[total_uniqueDesiredOutputsetCount];


    for (int i = 0; i < total_neuronCount; i++) { // setting each neuron's weight and bias to default-values
        weight(i) = 1;
        bias(i) = 0;
    }

    for (int recurrenceIndex = 0; recurrenceIndex <= total_recurrenceCount; recurrenceIndex++) {
        //sets to each neuron a ReLU-activated output-value
        for (int i = 1; i < total_layerCount; i++) { // skips to the next layer; starts at layer 1 (excl. input-layer which is layer 0)
            for (int j = layer_firstElementPointer(i); j < layer_firstElementPointer(i + 1); j++) { // deals with each neuron of the hidden and output layers
                if (j == layer_firstElementPointer(i + 1)) {
                    break;
                }
                float sum = 0;
                for (int k = layer_firstElementPointer(i - 1); k <= layer_firstElementPointer(i); k++) {
                    sum += output(k);
                }
                output(j) = defaultReLU(sum * weight[j] + bias[j]);
            }
        }

        // Softmax activation-function: transforms output-layer neurons into portions of a probability-distribution
        float zjSum = 0;
        for (int i = layer_firstElementPointer(total_layerCount - 1); i < total_neuronCount; i++) { // denominator
            zjSum += expf(output[i]);
        }
        for (int i = layer_firstElementPointer(total_layerCount - 1); i < total_neuronCount; i++) { // integration of denominator into the whole function
            output[i] = expf(output[i]) / zjSum; // Softmax completed
        }

        // error calculation using Mean Squared Error
        float mse = 0;
        for (int i = layer_firstElementPointer(total_layerCount - 1); i < outputLayer_neuronCount; i++) { // integration of denominator into the whole function
            mse += (output[i] - desiredOutput[i]) * (output[i] - desiredOutput[i]);
        }
        mse /= outputLayer_neuronCount;
        printf("Overall Mean-Squared-Error for iteration %d: %f", recurrenceIndex, mse);


        // weight-error-contribution tracing

        // bias-error-contribution tracing


        // gradiient-descent applied to weights and biases of each neuron
        for (int i = 0; i < total_neuronCount; i++) {
            weight(i) -= weightContributionToError[i];
            bias(i) -= biasContributionToError[i];
        }
    }







    // repeat over and over with swapped training input-values


    // evaluate the model using testing input-values

    // display the average Mean Squared Error across all tests, then end program


    return 0;
}