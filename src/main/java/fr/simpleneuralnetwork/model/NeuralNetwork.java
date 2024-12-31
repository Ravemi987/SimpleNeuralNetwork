package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.utils.MathsUtilities;


public class NeuralNetwork {

    private final Layer[] layers;

    public NeuralNetwork(int inputSize, int outputSize, int... neuronsPerLayer) {
        layers = new Layer[neuronsPerLayer.length + 2];
        InitLayers(inputSize, outputSize, neuronsPerLayer);
    }

    public void InitLayers(int inputSize, int outputSize, int[] neuronsPerLayer) {
        layers[0] = new Layer(inputSize, neuronsPerLayer[0]);

        for (int i = 1; i < neuronsPerLayer.length; i++) {
            layers[i] = new Layer(neuronsPerLayer[i - 1], neuronsPerLayer[i]);
        }

        layers[neuronsPerLayer.length + 1] = new Layer(neuronsPerLayer[neuronsPerLayer.length - 1], outputSize);
    }

    public double[] ForwardPropagation(double[] input) {
        for (Layer layer: layers) {
            input = layer.ForwardPropagation(input);
        }
        return input;
    }

    public double Loss(double[] input, double[] expectedOutputs) {
        double error = 0;
        double[] outputs = ForwardPropagation(input);
        Layer outputLayer = layers[layers.length - 1];

        for (int numOutput = 0; numOutput < outputs.length; numOutput++) {
            error += outputLayer.NeuronLoss(outputs[numOutput], expectedOutputs[numOutput]);
        }

        return error;
    }

    public double GlobalLoss(double[][] data, double[] expectedOutputs) {
        double totalError = 0;

        for (double[] input : data) {
            totalError += Loss(input, expectedOutputs);
        }

        return (1.0 / data.length) * totalError;
    }

    public void BackPropagation(double[] input, double[] expectedOutput) {
        double[] output = ForwardPropagation(input);
        Layer outputLayer = layers[layers.length - 1];
        double[] computedOutputGradients = outputLayer.ComputeOutputGradients(output, expectedOutput);

        for (int layer = layers.length - 2; layer >= 0; layer--) {
            computedOutputGradients = layers[layer].BackPropagation(layers[layer + 1], computedOutputGradients);
        }
    }

    public void UpdateAllWeights(double learningRate, int datasetSize) {
        for (Layer layer : layers) {
            layer.UpdateWeights(learningRate, datasetSize);
        }
    }

    public void BatchGradientDescent(double[][] trainInputs, double[][] expectedOutputs, double learningRate) {
        for (int input = 0; input < trainInputs.length; input++) {
            BackPropagation(trainInputs[input], expectedOutputs[input]);
        }
        UpdateAllWeights(learningRate, trainInputs.length);
    }

    public double[][] EncodeOutput(double[][] trainInputs, double[] expectedOutput) {
        int inputsNumber = trainInputs.length;
        int featuresNumber = trainInputs[0].length;
        double[][] expectedOutputs = new double[inputsNumber][featuresNumber];

        for (int input = 0; input < inputsNumber; input++) {
            int expectedIndex = (int) expectedOutput[input];
            expectedOutputs[input][expectedIndex] = 1.0;
        }

        return expectedOutputs;
    }

    public void Train(double[][] trainInputs, double[] expectedOutput, double learningRate,
                      double iterationsNumber, double decay) {
        double[][] expectedOutputs = EncodeOutput(trainInputs, expectedOutput);

        for (int epoch = 0; epoch < iterationsNumber; epoch++) {
            BatchGradientDescent(trainInputs, expectedOutputs, learningRate);
            learningRate = learningRate / (1 + decay * epoch);
        }
    }

    public double Predict(double[] testInput) {
        return MathsUtilities.MaxOfArray(ForwardPropagation(testInput));
    }

    public static double ActivationFunction(double z) {
        return MathsUtilities.Sigmoid(z);
    }

    public static double ActivationDerivative(double z) {
        return MathsUtilities.SigmoidDerivative(z);
    }
}
