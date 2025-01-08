package fr.simpleneuralnetwork.model;

import org.ejml.simple.SimpleMatrix;
import fr.simpleneuralnetwork.utils.MathsUtilities;

import java.util.Arrays;
import java.util.Random;

public class Layer {
    Random rand = new Random();

    private final int featuresNumber;
    private final int neuronsNumber;

    private double[][] activations;
    private double[][] linearInputs;

    private final double[][] weights;
    private final double[] biases;

    private final double[][] weightsGradients;
    private final double[] biasesGradients;

    public Layer(int nbFeatures, int nbNeurons) {
        this.featuresNumber = nbFeatures;
        this.neuronsNumber = nbNeurons;

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
        double[] output = new double[neuronsNumber];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            output[neuron] = MathsUtilities.Linear(input, weights[neuron], biases[neuron]);
            output[neuron] = Forward(output[neuron]);
        }
        return output;
    }

    public double[][] ForwardPropagationBatch(double[][] inputs) {
        int batchSize = inputs.length;

        this.activations = new double[batchSize][featuresNumber];
        this.linearInputs = new double[batchSize][neuronsNumber];

        SaveActivations(inputs, batchSize);
        ComputeMatrixOperations(inputs, batchSize);


        return MathsUtilities.ApplyActivation(linearInputs, this::Forward);
    }

    public void SaveActivations(double[][] inputs, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(inputs[i], 0, activations[i], 0, featuresNumber);
        }
    }

    public void ComputeMatrixOperations(double[][] inputs, int batchSize) {
        SimpleMatrix inputMatrix = new SimpleMatrix(inputs);
        SimpleMatrix weightMatrix = new SimpleMatrix(weights).transpose();
        SimpleMatrix biasMatrix = new SimpleMatrix(1, biases.length, true, biases);

        SimpleMatrix extendedBiasMatrix = new SimpleMatrix(batchSize, biases.length);
        for (int i = 0; i < batchSize; i++) {
            extendedBiasMatrix.insertIntoThis(i, 0, biasMatrix);
        }

        SimpleMatrix prodMatrix = inputMatrix.mult(weightMatrix);
        SimpleMatrix linearInputMatrix = prodMatrix.plus(extendedBiasMatrix);

        this.linearInputs = linearInputMatrix.getDDRM().get2DData();
    }

    public void InitWeights() {
        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            biases[neuron] = rand.nextDouble() * 2 - 1;

            for (int feature = 0; feature < featuresNumber; feature++) {
                weights[neuron][feature] = rand.nextDouble() * 2 - 1;
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

    public void UpdateWeightsGradients(int neuron, double nextGradientValue, int batchIndex) {
        for (int feature = 0; feature < featuresNumber; feature++) {
            weightsGradients[neuron][feature] += activations[batchIndex][feature] * nextGradientValue;
        }
    }

    public double[][] ComputeOutputGradientsBatch(double[][] outputs, double[][] expectedOutputs) {
        int batchSize = outputs.length;
        double[][] gradients = new double[batchSize][neuronsNumber];

        for (int i = 0; i < batchSize; i++) {
            gradients[i] = ComputeOutputGradients(outputs, expectedOutputs, i);
        }
        return  gradients;
    }

    public double[] ComputeOutputGradients(double[][] outputs, double[][] expectedOutputs, int batchIndex) {
        double[] derivativeLossOutput = new double[outputs[0].length];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            double dLoss = LossDerivative(outputs[batchIndex][neuron], expectedOutputs[batchIndex][neuron]);
            double forwardedDerivative = ForwardDerivative(linearInputs[batchIndex][neuron]);
            derivativeLossOutput[neuron] = dLoss * forwardedDerivative;

            UpdateWeightsGradients(neuron, derivativeLossOutput[neuron], batchIndex);
            biasesGradients[neuron] += derivativeLossOutput[neuron];
        }
        return derivativeLossOutput;
    }

    public double[][] BackPropagationBatch(Layer nextLayer, double[][] nextGradients) {
        int batchSize = nextGradients.length;
        double[][] gradients = new double[batchSize][neuronsNumber];

        for (int i = 0; i < batchSize; i++) {
            gradients[i] = BackPropagation(nextLayer, nextGradients, i);
        }
        return gradients;
    }

    public double[] BackPropagation(Layer nextLayer, double[][] nextGradients, int batchIndex) {
        double[] currentGradient = new double[neuronsNumber];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            currentGradient[neuron] = 0;
            double forwardedDerivative = ForwardDerivative(linearInputs[batchIndex][neuron]);

            for (int feature = 0; feature < nextLayer.getNeuronsNumber(); feature++) {
                double connectionWeight = nextLayer.getWeights()[feature][neuron];
                currentGradient[neuron] += connectionWeight * nextGradients[batchIndex][feature];
            }
            currentGradient[neuron] *= forwardedDerivative;
            UpdateWeightsGradients(neuron, currentGradient[neuron], batchIndex);
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
