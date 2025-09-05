#include <stdio.h>
#include <math.h>

// ReLU function
float defaultReLU(float input) {
    if (input <= 0) {
        return 0; // x < 0 --> max(0, x) = 0
    }
    else {
        return input; // x > 0 --> max(0, x) = x
    }
}

// first derivative of ReLU function
float derivative1ReLU(float input) {
    if (defaultReLU(input) != 0) {
        return 1; // x' = 1
    }
    else {
        return 0; // 0' = 0
    }
}

int main(void) {
    printf("Welcome to my artificial neural network!");

    int total_neuronCount;
    int total_layerCount;
    int total_uniqueDesiredOutputsetCount;

    int layer_firstElementPointer[total_layerCount]; // all layers incl. the input layer and incl. the position immediately after total_neuronCount
    int outputLayer_neuronCount = total_neuronCount - layer_firstElementPointer(total_layerCount - 1);

    float weight[total_neuronCount];
    float bias[total_neuronCount];
    float output[total_neuronCount];

    float desiredOutput[outputLayer_neuronCount * total_uniqueDesiredOutputsetCount];
    int desiredOutput_firstElementPointer[total_uniqueDesiredOutputsetCount];


    for (int i = 0; i < total_neuronCount; i++) { // setting each neuron's weight and bias to default-values
        weight(i) = 1;
        bias(i) = 0;
    }

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

    for (int i = layer_firstElementPointer[total_layerCount - 1] - 1; i >= layer_firstElementPointer[total_layerCount - 2]; i--) {
        float sum1 = 0;
        for (int j = total_neuronCount - 1; j >= layer_firstElementPointer[total_layerCount]; j--) {

        }

        float sum2 = 0;
        for (int m = 0; m < total_uniqueDesiredOutputsetCount; m++) {
            for (int j = total_neuronCount - 1; j >= layer_firstElementPointer[total_layerCount]; j--) {
                sum2 += output[i] + 2 * (weight[j] * derivative1ReLU() * defaultReLU() - desiredOutput[j]);
            }
        }
        output[i] += sum2 / total_uniqueDesiredOutputsetCount;
    }
    if (total_layerCount > 3) {
        for (int n = 0; n < total_layerCount; n++) {
            for (int i = layer_firstElementPointer[layer_firstElementPointer - n] - 1; i >= layer_firstElementPointer[total_layerCount - 2]; i--) {
                float sum1 = 0;
                for (int j = total_neuronCount - 1; j >= layer_firstElementPointer[total_layerCount]; j--) {

                }

                float sum2 = 0;
                for (int j = total_neuronCount - 1; j >= layer_firstElementPointer[total_layerCount]; j--) {
                    sum2 += output[i] + 2 * (weight[j] * derivative1ReLU() * defaultReLU() - desiredOutput[j]);
                }
                output[i] += sum2;
            }
        }
    }




    // repeat over and over with swapped training input-values


    // evaluate the model using testing input-values

    // display the average Mean Squared Error across all tests, then end program


    return 0;
}