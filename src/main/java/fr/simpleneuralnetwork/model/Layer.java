package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.utils.MathsUtilities;

import java.util.Random;

public class Layer {
    Random rand = new Random();

    private final int featuresNumber;
    private final int neuronsNumber;

    private final double[] previousActivations;
    private final double[] linearInputs;

    private final double[][] weights;
    private final double[] biases;

    private final double[][] weightsGradients;
    private final double[] biasesGradients;

    public Layer(int nbFeatures, int nbNeurons) {
        this.featuresNumber = nbFeatures;
        this.neuronsNumber = nbNeurons;

        previousActivations = new double[featuresNumber];
        linearInputs = new double[neuronsNumber];

        weights = new double[neuronsNumber][featuresNumber];
        biases = new double[neuronsNumber];

        weightsGradients = new double[neuronsNumber][featuresNumber];
        biasesGradients = new double[neuronsNumber];

        InitWeights();
    }

    // Test
    public Layer(int nbFeatures, int nbNeurons, double[][] initialWeights, double[] initialBiases) {
        this.featuresNumber = nbFeatures;
        this.neuronsNumber = nbNeurons;
        this.weights = initialWeights;
        this.biases = initialBiases;

        previousActivations = new double[featuresNumber];
        linearInputs = new double[neuronsNumber];
        weightsGradients = new double[neuronsNumber][featuresNumber];
        biasesGradients = new double[neuronsNumber];
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public int getFeaturesNumber() {
        return featuresNumber;
    }

    public int getNeuronsNumber() {
        return neuronsNumber;
    }

    public double Forward(double z) {
        return NeuralNetwork.ActivationFunction(z);
    }

    public double ForwardDerivative(double z) {
        return NeuralNetwork.ActivationDerivative(z);
    }

    public double[] ForwardPropagation(double[] input) {
        double[] outputs = new double[neuronsNumber];

        if (featuresNumber >= 0) System.arraycopy(input, 0, previousActivations, 0, featuresNumber);

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            linearInputs[neuron] = MathsUtilities.Linear(input, weights[neuron], biases[neuron]);
            outputs[neuron] = Forward(linearInputs[neuron]);
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

    public void UpdateWeightsGradients(int neuron, double nextGradientValue) {
        for (int feature = 0; feature < featuresNumber; feature++) {
            weightsGradients[neuron][feature] += previousActivations[feature] * nextGradientValue;
        }
    }

    public double[] ComputeOutputGradients(double[] output, double[] expectedOutput) {
        double[] derivativeLossOutput = new double[output.length];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            double dLoss = LossDerivative(output[neuron], expectedOutput[neuron]);
            double forwardedDerivative = ForwardDerivative(linearInputs[neuron]);
            derivativeLossOutput[neuron] = dLoss * forwardedDerivative;

            UpdateWeightsGradients(neuron, derivativeLossOutput[neuron]);
            biasesGradients[neuron] += derivativeLossOutput[neuron];
        }
        return derivativeLossOutput;
    }

    public double[] BackPropagation(Layer nextLayer, double[] nextGradient) {
        double[] currentGradient = new double[neuronsNumber];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            currentGradient[neuron] = 0;
            double forwardedDerivative = ForwardDerivative(linearInputs[neuron]);

            for (int feature = 0; feature < nextGradient.length; feature++) {
                double connectionWeight = nextLayer.getWeights()[feature][neuron];
                currentGradient[neuron] += connectionWeight * nextGradient[feature];
            }
            UpdateWeightsGradients(neuron, currentGradient[neuron]);
            currentGradient[neuron] *= forwardedDerivative;
            biasesGradients[neuron] += currentGradient[neuron];
        }
        return currentGradient;
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
