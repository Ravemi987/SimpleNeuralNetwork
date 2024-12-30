package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.utils.MathsUtilities;

import java.util.Random;

public class Layer {
    Random rand = new Random();

    private final int featuresNumber;
    private final int neuronsNumber;

    //private double[] activations;
    private final double[] linearInputs;

    private final double[][] weights;
    private final double[] biases;

    private final double[][] weightsGradients;
    private final double[] biasesGradients;

    public Layer(int nbFeatures, int nbNeurons) {
        this.featuresNumber = nbFeatures;
        this.neuronsNumber = nbNeurons;

        //activations = new double[neuronsNumber];
        linearInputs = new double[neuronsNumber];

        weights = new double[neuronsNumber][featuresNumber];
        biases = new double[neuronsNumber];

        weightsGradients = new double[neuronsNumber][featuresNumber];
        biasesGradients = new double[neuronsNumber];

        InitWeights();
    }

    public double[][] getWeights() {
        return weights;
    }

    public double Forward(double z) {
        return NeuralNetwork.ActivationFunction(z);
    }

    public double ForwardDerivative(double z) {
        return NeuralNetwork.ActivationDerivative(z);
    }

    public double[] ForwardPropagation(double[] input) {
        double[] outputs = new double[neuronsNumber];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            linearInputs[neuron] = MathsUtilities.Linear(input, weights[neuron], biases[neuron]);
            outputs[neuron] = Forward(linearInputs[neuron]);
            //activations[neuron] = outputs[neuron];
        }

        return outputs;
    }

    public void InitWeights() {
        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            biases[neuron] = 2 * rand.nextDouble() - 1;

            for (int feature = 0; feature < featuresNumber; feature++) {
                weights[neuron][feature] = 2 * rand.nextDouble() - 1;
            }
        }
    }

    public double NeuronLoss(double output, double expectedOutput) {
        double diff = output - expectedOutput;
        return diff * diff;
    }

    public double LossDerivative(double output, double expectedOutput) {
        return 2 * (output - expectedOutput);
    }

    public double[] ComputeOutputGradients(double[] input, double[] output, double[] expectedOutput) {
        double[] derivativeLossOutput = new double[output.length];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            double dLoss = LossDerivative(output[neuron], expectedOutput[neuron]);
            double forwardedDerivative = ForwardDerivative(linearInputs[neuron]);
            derivativeLossOutput[neuron] = dLoss * forwardedDerivative;

            for (int feature = 0; feature < featuresNumber; feature++) {
                weightsGradients[neuron][feature] += input[feature] * derivativeLossOutput[neuron];
            }
            biasesGradients[neuron] += derivativeLossOutput[neuron];
        }
        return derivativeLossOutput;
    }

    public double[] BackPropagation(Layer nextLayer, double[] input, double[] nextGradient) {
        double[] currentGradient = new double[featuresNumber];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            currentGradient[neuron] = 0;
            double forwardedDerivative = ForwardDerivative(linearInputs[neuron]);

            for (int feature = 0; feature < featuresNumber; feature++) {
                double connectionWeight = nextLayer.getWeights()[neuron][feature];
                currentGradient[neuron] += connectionWeight * nextGradient[neuron];
                weightsGradients[neuron][feature] += input[feature] * currentGradient[neuron];
            }
            currentGradient[neuron] *= forwardedDerivative;
            biasesGradients[neuron] += currentGradient[neuron];
        }
        return nextGradient;
    }

    public void UpdateWeights(double learningRate, int datasetSize) {
        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            for (int feature = 0; feature < featuresNumber; feature++) {
                weights[neuron][feature] -= learningRate * weightsGradients[neuron][feature] / datasetSize;
                weightsGradients[neuron][feature] = 0;
            }
            biases[neuron] -= learningRate * biasesGradients[neuron] / datasetSize;
            biasesGradients[neuron] = 0;
        }
    }
}
