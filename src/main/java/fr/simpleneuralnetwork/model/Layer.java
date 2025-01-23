package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.model.Activations.Sigmoid;
import fr.simpleneuralnetwork.model.Activations.SoftMax;
import fr.simpleneuralnetwork.model.Losses.CrossEntropy;
import fr.simpleneuralnetwork.model.Losses.MeanSquaredError;
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

    private IActivation activationFunction;

    public Layer(int nbFeatures, int nbNeurons, String activationFun) {
        this.featuresNumber = nbFeatures;
        this.neuronsNumber = nbNeurons;
        ScanActivationFunction(activationFun);

        weights = new double[neuronsNumber][featuresNumber];
        biases = new double[neuronsNumber];

        weightsGradients = new double[neuronsNumber][featuresNumber];
        biasesGradients = new double[neuronsNumber];

        InitWeights();
    }

    public Layer(double[][] initialWeights, double[] initialBiases, int nbFeatures, int nbNeurons, String activationFun) {
        this.featuresNumber = nbFeatures;
        this.neuronsNumber = nbNeurons;
        this.weights = initialWeights;
        this.biases = initialBiases;
        ScanActivationFunction(activationFun);

        weightsGradients = new double[neuronsNumber][featuresNumber];
        biasesGradients = new double[neuronsNumber];
    }

    public void ScanActivationFunction(String activationFun) {
        switch(activationFun) {
            case "Sigmoid":
                activationFunction = new Sigmoid();
                break;
            case "Softmax":
                activationFunction = new SoftMax();
                break;
            default:
                System.err.println("Unknown loss function.");
                System.exit(-1);
        }
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

    public double[][] ForwardPropagation(double[][] inputs) {
        this.linearInputs = new double[1][neuronsNumber];
        ForwardLinear(inputs, 1);

        return activationFunction.ApplyMatrix(linearInputs);
    }

    public double[][] ForwardPropagationBatch(double[][] inputs) {
        int batchSize = inputs.length;

        this.activations = new double[batchSize][featuresNumber];
        this.linearInputs = new double[batchSize][neuronsNumber];

        SaveActivations(inputs, batchSize);
        ForwardLinear(inputs, batchSize);

        return activationFunction.ApplyMatrix(linearInputs);
    }

    public void SaveActivations(double[][] inputs, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(inputs[i], 0, activations[i], 0, featuresNumber);
        }
    }

    public void ForwardLinear(double[][] inputs, int batchSize) {
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

    public void UpdateWeightsGradients(int neuron, double nextGradientValue, int batchIndex) {
        for (int feature = 0; feature < featuresNumber; feature++) {
            weightsGradients[neuron][feature] += activations[batchIndex][feature] * nextGradientValue;
        }
    }

    public double[][] ComputeOutputGradientsBatch(double[][] outputs, double[][] expectedOutputs) {
        int batchSize = outputs.length;
        double[][] gradients = new double[batchSize][neuronsNumber];
        double[][] lossDerivatives = NeuralNetwork.getLossFunction().DerivativeMatrix(outputs, expectedOutputs);
        double[][] forwardedDerivatives = activationFunction.DerivativeMatrix(linearInputs);

        for (int i = 0; i < batchSize; i++) {
            gradients[i] = ComputeOutputGradients(lossDerivatives, forwardedDerivatives, i);
        }
        return  gradients;
    }

    public double[] ComputeOutputGradients(double[][] lossDerivatives, double[][] forwardedDerivatives, int batchIndex) {
        double[] derivativeLossOutput = new double[lossDerivatives[0].length];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            derivativeLossOutput[neuron] = lossDerivatives[batchIndex][neuron] * forwardedDerivatives[batchIndex][neuron];

            UpdateWeightsGradients(neuron, derivativeLossOutput[neuron], batchIndex);
            biasesGradients[neuron] += derivativeLossOutput[neuron];
        }
        return derivativeLossOutput;
    }

    public double[][] BackPropagationBatch(Layer nextLayer, double[][] nextGradients) {
        int batchSize = nextGradients.length;
        double[][] gradients = new double[batchSize][neuronsNumber];
        double[][] forwardedDerivatives = activationFunction.DerivativeMatrix(linearInputs);

        for (int i = 0; i < batchSize; i++) {
            gradients[i] = BackPropagation(nextLayer, nextGradients, forwardedDerivatives, i);
        }
        return gradients;
    }

    public double[] BackPropagation(Layer nextLayer, double[][] nextGradients,
                                    double[][] forwardedDerivatives, int batchIndex) {
        double[] currentGradient = new double[neuronsNumber];

        for (int neuron = 0; neuron < neuronsNumber; neuron++) {
            currentGradient[neuron] = 0;

            for (int feature = 0; feature < nextLayer.getNeuronsNumber(); feature++) {
                double connectionWeight = nextLayer.getWeights()[feature][neuron];
                currentGradient[neuron] += connectionWeight * nextGradients[batchIndex][feature];
            }
            currentGradient[neuron] *= forwardedDerivatives[batchIndex][neuron];
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
