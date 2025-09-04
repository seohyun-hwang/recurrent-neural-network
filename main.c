#include <stdio.h>
#include <math.h>

int main(void) {
    printf("Welcome to my artificial neural network!");

    int total_neuronCount;
    int total_layerCount;

    int layer_firstElementPointer[total_layerCount]; // all layers incl. the input layer and incl. the position immediately after total_neuronCount
    int outputLayer_neuronCount = total_neuronCount - layer_firstElementPointer(total_layerCount - 1);

    float weight[total_neuronCount];
    float bias[total_neuronCount];
    float output[total_neuronCount];

    float desiredOutput[outputLayer_neuronCount];


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

            // ReLU activation-function
            float output = sum * weight[j] + bias[j];
            if (output < 0) {
                output(j) = 0;
            }
            else {
                output(j) = output;
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

    // error calculation using Mean Squared Error
    float mse = 0;
    for (int i = layer_firstElementPointer(total_layerCount - 1); i < outputLayer_neuronCount; i++) { // integration of denominator into the whole function
        mse += (output[i] - desiredOutput[i]) * (output[i] - desiredOutput[i]);
    }
    mse /= outputLayer_neuronCount;


    // computing the negative error gradient through backpropagation chain rule

    // float errorGradient = 2 * currentLayer_effectSumWrtPrevOutput
    // float currentLayer_effectSum = wjL * ReLU'(zjL) * (ReLU(zjL) - yj)

    // computing the gradient descent and adjusting weights/biases accordingly

    // previousLayerNeuron_desiredOutput = ak - currentLayer_effectSumWrtPrevOutput
    // ak = output(i - 1)?



    // repeat over and over with swapped training input-values


    // evaluate the model using testing input-values

    // display the average Mean Squared Error across all tests, then end program


    return 0;
}