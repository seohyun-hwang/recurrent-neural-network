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


    // not recurring!
    for (int i = 0; i < total_neuronCount; i++) { // setting each neuron's weight and bias to default-values
        weight(i) = 1;
        bias(i) = 0;
    }

    // recurrence starts here
    for (int recurrenceIndex = 0; recurrenceIndex <= total_recurrenceCount; recurrenceIndex++) {
        //sets to each neuron a ReLU-activated output-value
        for (int i = 1; i < total_layerCount; i++) { // skips to the next layer; starts at layer 1 (excl. input-layer which is layer 0)
            for (int j = layer_firstElementPointer(i); j < layer_firstElementPointer(i + 1); j++) { // deals with each neuron of the hidden and output layers
                if (j == layer_firstElementPointer(i + 1)) {
                    break;
                }
                float sum = 0;
                for (int k = layer_firstElementPointer(i - 1); k <= layer_firstElementPointer(i); k++) { // sum of previous-layer's outputs
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

        // error calculation using cross-entropy loss
        float cel = 0;
        for (int i = layer_firstElementPointer(total_layerCount - 1); i < outputLayer_neuronCount; i++) {
            cel += desiredOutput[i] * logf(output[i]);
        }
        cel *= -1;
        printf("Overall Cross-Entropy Loss for iteration %d = %f", recurrenceIndex, cel);


        // error-contribution tracing (weight and bias simultaneously for each neuron) with gradient-descent implementation
        float dC_dYhat = 0; // calculating the last term of d(C)/d(param)
        for (int i = layer_firstElementPointer(total_layerCount - 1); i < outputLayer_neuronCount; i++) {
            dC_dYhat += desiredOutput[i] * output[i];
        }
        dC_dYhat *= -1;

        for (int i = 0; i < total_neuronCount; i++) { // this should traverse the other way around
            float sum = 0;
            for (int k = layer_firstElementPointer(i - 1); k <= layer_firstElementPointer(i); k++) { // sum of previous-layer's outputs
                sum += output(k);
            }

            weight[i] -= sum * derivative1ReLU(bias[i] + weight[i] * sum) * dC_dYhat;
            bias[i] -= derivative1ReLU(bias[i] + weight[i] * sum) * dC_dYhat;
        }
    }

    return 0;
}