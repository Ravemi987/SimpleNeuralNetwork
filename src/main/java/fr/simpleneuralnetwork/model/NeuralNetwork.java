package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.utils.MathsUtilities;

import java.util.Arrays;


public class NeuralNetwork {

    private final Layer[] layers;

    public NeuralNetwork(int... layerSizes) {
        layers = new Layer[layerSizes.length - 1];
        InitLayers(layerSizes);
    }

    public void InitLayers(int[] layerSizes) {
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }
    }

    // Test
    public NeuralNetwork(double[][][] initialWeights, double[][] initialBiases, int[] layerSizes) {
        layers = new Layer[layerSizes.length - 1];

        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(
                    layerSizes[i],
                    layerSizes[i + 1],
                    initialWeights[i],
                    initialBiases[i]
            );
        }
    }

    public double[] ForwardPropagation(double[] input) {
        for (Layer layer: layers) {
            input = layer.ForwardPropagation(input);
        }
        return input;
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

    public double[][] OneHotEncoder(double[] expectedOutput, int numClasses) {
        int inputsNumber = expectedOutput.length;
        double[][] encodedOutputs = new double[inputsNumber][numClasses];

        for (int i = 0; i < inputsNumber; i++) {
            int expectedIndex = (int) expectedOutput[i];
            encodedOutputs[i][expectedIndex] = 1.0;
        }

        return encodedOutputs;
    }

    public void Train(double[][] trainInputs, double[] expectedOutput, double learningRate,
                      double iterationsNumber, double decay) {
        double[][] expectedOutputs = OneHotEncoder(expectedOutput, 2);
        DisplayExpectedOutputs(expectedOutput);
        DisplayEncodedExpectedOutputs(expectedOutputs);

        for (int epoch = 0; epoch <= iterationsNumber; epoch++) {
            BatchGradientDescent(trainInputs, expectedOutputs, learningRate);
            learningRate = learningRate / (1 + decay * epoch);
        }
        System.out.println("Loss " + " " + GlobalLoss(trainInputs, expectedOutputs));
        DisplayTrainAccuracy(trainInputs, expectedOutputs);
    }

    public double[] Predict(double[] testInput) {
        return ForwardPropagation(testInput);
    }

    public double[][] PredictProba(double[][] testInputs) {
        double[][] predictions = new double[testInputs.length][];

        for (int i = 0; i < predictions.length; i++) {
            predictions[i] = ForwardPropagation(testInputs[i]);
        }

        return predictions;
    }

    public double[] PredictClasses(double[][] testInputs) {
        double[] predictions = new double[testInputs.length];

        for (int i = 0; i < predictions.length; i++) {
            double[] outputs = ForwardPropagation(testInputs[i]);

            if (outputs.length == 1) {
                predictions[i] = outputs[0] >= 0.5 ? 0.0 : 1.0;
            } else {
                predictions[i] = MathsUtilities.IndexMaxOfArray(outputs);
            }
        }

        return predictions;
    }

    public double GetAccuracy(double[] predictions, double[][] testLabels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] == MathsUtilities.IndexMaxOfArray(testLabels[i])) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }

    public static double ActivationFunction(double z) {
        return MathsUtilities.Sigmoid(z);
    }

    public static double ActivationDerivative(double z) {
        return MathsUtilities.SigmoidDerivative(z);
    }

    // ***************** HELPERS *****************

    public void DisplayExpectedOutputs(double[] expectedOutput) {
        System.out.println("Expected outputs: ");
        System.out.println(Arrays.toString(expectedOutput));
    }

    public void DisplayEncodedExpectedOutputs(double[][] expectedOutputs) {
        System.out.println("Encoded expected outputs: ");

        for (double[] output: expectedOutputs) {
            System.out.println(Arrays.toString(output));
        }
    }

    public void DisplayPredictions(double[][] predictions) {
        for (int i = 0; i < predictions.length; i++) {
            System.out.println(i + "    " + Arrays.toString(predictions[i]));
        }
    }

    public void DisplayTrainAccuracy(double[][] trainInputs, double[][] expectedOutputs) {
        System.out.println("accuracy: " + GetAccuracy(PredictClasses(trainInputs), expectedOutputs));
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

    public double GlobalLoss(double[][] inputs, double[][] expectedOutputs) {
        double totalError = 0;

        for (int i = 0; i < inputs.length; i++) {
            totalError += Loss(inputs[i], expectedOutputs[i]);
        }

        return (1.0 / inputs.length) * totalError;
    }

    public int getWeightsNumber() {
        int weightsNumber = 0;

        for (Layer layer: layers) {
            int nbNeurons = layer.getNeuronsNumber();
            weightsNumber += nbNeurons * (1 + layer.getFeaturesNumber());
        }

        return weightsNumber;
    }

    public double[] GetAllWeights() {
        int weightsNumber = getWeightsNumber();
        double[] allWeights = new double[weightsNumber];
        int index = 0;

        for (Layer layer : layers) {
            for (int neuron = 0; neuron < layer.getNeuronsNumber(); neuron++) {
                for (int feature = 0; feature < layer.getFeaturesNumber(); feature++) {
                    allWeights[index++] = layer.getWeights()[neuron][feature];
                }
                allWeights[index++] = layer.getBiases()[neuron];
            }
        }

        return allWeights;
    }
}
