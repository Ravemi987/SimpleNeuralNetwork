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

    private double[][] weightsGradients;
    private double[] biasesGradients;

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
        ComputeMatrixForward(inputs, batchSize);

        return MathsUtilities.ApplyActivation(linearInputs, this::Forward);
    }

    public void SaveActivations(double[][] inputs, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(inputs[i], 0, activations[i], 0, featuresNumber);
        }
    }

    public void ComputeMatrixForward(double[][] inputs, int batchSize) {
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

    public void UpdateGradients(SimpleMatrix newGradientsMatrix) {
        SimpleMatrix weightsGradientsMatrix = newGradientsMatrix.transpose().mult(new SimpleMatrix(activations)); // [neuronsCurrent * featuresCurrent]
        weightsGradients = weightsGradientsMatrix.getDDRM().get2DData();

        double[] tmpArr = new double[newGradientsMatrix.getNumRows()];
        Arrays.fill(tmpArr, 1.0);

        SimpleMatrix biasesGradientsMatrix = newGradientsMatrix.transpose().mult(new SimpleMatrix(newGradientsMatrix.getNumRows(),
                1, true, tmpArr)
        ); // [neuronsCurrent * 1]

        biasesGradients = biasesGradientsMatrix.getDDRM().getData();
    }

    public double[][] ComputeOutputGradientsBatch(double[][] outputs, double[][] expectedOutputs) {
        SimpleMatrix forwardedDerivatives = new SimpleMatrix(
                MathsUtilities.ApplyActivation(linearInputs, this::ForwardDerivative) // [batchSize * neuronsCurrent]
        );
        SimpleMatrix lossDerivatives = new SimpleMatrix(
                MathsUtilities.ApplyLoss(outputs, expectedOutputs, this::LossDerivative) // [batchSize * neuronsCurrent]
        );
        SimpleMatrix newGradientsMatrix = lossDerivatives.elementMult(forwardedDerivatives);
        UpdateGradients(newGradientsMatrix);

        return newGradientsMatrix.getDDRM().get2DData();
    }

    public double[][] BackPropagationBatch(Layer nextLayer, double[][] nextGradients) {
        SimpleMatrix forwardedDerivatives = new SimpleMatrix(
                MathsUtilities.ApplyActivation(linearInputs, this::ForwardDerivative) // [batchSize * neuronsCurrent]
        );
        SimpleMatrix weightsMatrix= new SimpleMatrix(nextLayer.getWeights()); // [neuronsNext x neuronsCurrent]
        SimpleMatrix nextGradientsMatrix = new SimpleMatrix(nextGradients); // [batchSize * neuronsNext]
        SimpleMatrix newGradientsMatrix = nextGradientsMatrix.mult(weightsMatrix).elementMult(forwardedDerivatives); // [batchSize * neuronsCurrent]
        UpdateGradients(newGradientsMatrix);

        return newGradientsMatrix.getDDRM().get2DData();
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
