#include <stdio.h>
#include <math.h>

int main(void) {
    printf("Welcome to my artificial neural network!");

    int total_neuronCount;
    int total_layerCount;

    int layer_firstElementPointer[total_layerCount]; // all layers incl. the input layer and incl. the position immediately after total_neuronCount

    float weight[total_neuronCount];
    float bias[total_neuronCount];
    float output[total_neuronCount];


    for (int i = 0; i < total_neuronCount; i++) { // setting each neuron's weight and bias to default
        weight(i) = 1;
        bias(i) = 0;
    }

    //sets to each neuron a nonactivated output-value
    for (int i = 1; i < total_layerCount; i++) { // skips to the next layer; starts at layer 1 (excl. input-layer which is layer 0)
        for (int j = layer_firstElementPointer(i); j < layer_firstElementPointer(i + 1); j++) { // looks at each neuron
            if (j == layer_firstElementPointer(i + 1)) {
                break;
            }
            float sum = 0;
            for (int k = layer_firstElementPointer(i - 1); k <= layer_firstElementPointer(i); k++) {
                sum += output(k);
            }

            // ReLU activation-function
            if (sum * weight[j] + bias[j] < 0) {
                output[j] = 0;
            }
            else {
                output[j] = sum * weight[j] + bias[j];
            }
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

    // loss check


    return 0;
}