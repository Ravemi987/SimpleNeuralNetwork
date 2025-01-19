package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.model.Losses.CrossEntropy;
import fr.simpleneuralnetwork.model.Losses.MeanSquaredError;
import fr.simpleneuralnetwork.utils.MathsUtilities;

import java.util.Arrays;


public class NeuralNetwork {

    private final Layer[] layers;
    private static ILoss lossFunction;

    public NeuralNetwork(String loss, String hiddenActivation, String outputActivation, int... layerSizes) {
        layers = new Layer[layerSizes.length - 1];
        ScanLossFunction(loss);
        InitLayers(layerSizes, hiddenActivation, outputActivation);
    }

    public void ScanLossFunction(String loss) {
        switch(loss) {
            case "MeanSquaredError":
              lossFunction = new MeanSquaredError();
              break;
            case "CrossEntropy":
                lossFunction = new CrossEntropy();
                break;
            default:
                System.err.println("Unknown loss function.");
                System.exit(-1);
        }
    }

    public void InitLayers(int[] layerSizes, String hiddenActivation, String outputActivation) {
        for (int i = 0; i < layers.length; i++) {
            if (i < layers.length - 1) {
                layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], hiddenActivation);
            } else {
                layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], outputActivation);
            }
        }
    }

    // Test
    public NeuralNetwork(double[][][] initialWeights, double[][] initialBiases, int[] layerSizes,
                         String loss, String hiddenActivation, String outputActivation) {
        layers = new Layer[layerSizes.length - 1];
        ScanLossFunction(loss);

        for (int i = 0; i < layers.length; i++) {
            if (i < layers.length - 1) {
                layers[i] = new Layer(
                        layerSizes[i],
                        layerSizes[i + 1],
                        initialWeights[i],
                        initialBiases[i],
                        hiddenActivation
                );
            } else {
                layers[i] = new Layer(
                        layerSizes[i],
                        layerSizes[i + 1],
                        initialWeights[i],
                        initialBiases[i],
                        outputActivation
                );
            }

        }
    }

    public double[] NNForwardPropagation(double[] input) {
        double[][] activations = {input};

        for (Layer layer: layers) {
            activations = layer.ForwardPropagation(activations);
        }
        return activations[0];
    }

    public double[][] NNForwardPropagationBatch(double[][] inputs) {
        double[][] activations = inputs;

        for (Layer layer: layers) {
            activations = layer.ForwardPropagationBatch(activations);
        }
        return activations;
    }

    public void BackPropagation(double[][] outputs, double[][] expectedOutputs) {
        Layer outputLayer = layers[layers.length - 1];
        double[][] computedOutputGradients = outputLayer.ComputeOutputGradientsBatch(outputs, expectedOutputs);

        for (int layer = layers.length - 2; layer >= 0; layer--) {
            computedOutputGradients = layers[layer].BackPropagationBatch(layers[layer + 1], computedOutputGradients);
        }
    }

    public void UpdateAllWeights(double learningRate, int datasetSize) {
        for (Layer layer : layers) {
            layer.UpdateWeights(learningRate, datasetSize);
        }
    }

    public void BatchGradientDescent(double[][] trainInputs, double[][] expectedOutputs,
                                     double learningRate, int batchSize) {
        int totalSize = trainInputs.length;
        int batchesNumber = (int) Math.ceil((double) totalSize / batchSize);
        double totalLoss = 0;
        int totalCorrect = 0;

        for (int batch = 0; batch < batchesNumber; batch++) {
            int start = batch * batchSize;
            int end = Math.min(start + batchSize, totalSize);

            double[][] batchInputs = Arrays.copyOfRange(trainInputs, start, end);
            double[][] batchOutputs = Arrays.copyOfRange(expectedOutputs, start, end);
            double[][] outputs = NNForwardPropagationBatch(batchInputs);

            BackPropagation(outputs, batchOutputs);

            totalLoss += lossFunction.GlobalLoss(outputs, batchOutputs);
            totalCorrect += GetCorrectPredictions(outputs, batchOutputs);

            UpdateAllWeights(learningRate, batchSize);
        }

        double averageLoss = totalLoss / batchesNumber;
        double accuracy = (totalCorrect / (double) totalSize) * 100;
        System.out.printf("Loss: %.6f - Accuracy: %.2f%%%n", averageLoss, accuracy);
    }

    public void Train(double[][] trainInputs, double[] expectedOutput, double learningRate,
                      double iterationsNumber, int batchSize, double decay) {
        double[][] expectedOutputs = OneHotEncoder(expectedOutput, trainInputs[0].length);
        double initialLr = learningRate;

        for (int epoch = 0; epoch <= iterationsNumber; epoch++) {
            System.out.print("Epoch " + epoch + " - ");
            BatchGradientDescent(trainInputs, expectedOutputs, learningRate, batchSize);
            learningRate = initialLr / (1 + decay * epoch);
        }
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

    public int GetCorrectPredictions(double[][] predictions, double[][] expectedOutputs) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            int predictedClass = MathsUtilities.IndexMaxOfArray(predictions[i]);
            int expectedClass = MathsUtilities.IndexMaxOfArray(expectedOutputs[i]);

            if (predictedClass == expectedClass ) {
                correct++;
            }
        }
        return correct;
    }

    public double[] Predict(double[] testInput) {
        return NNForwardPropagation(testInput);
    }

    public double[][] PredictAll(double[][] testInputs) {
        double[][] predictions = new double[testInputs.length][];

        for (int i = 0; i < predictions.length; i++) {
            predictions[i] = NNForwardPropagation(testInputs[i]);
        }

        return predictions;
    }

    public double PredictClass(double[] testInput) {
        double[] outputs = NNForwardPropagation(testInput);
        if (outputs.length == 1) {
            return outputs[0] >= 0.5 ? 0.0 : 1.0;
        } else {
            return MathsUtilities.IndexMaxOfArray(outputs);
        }
    }

    public double[] PredictAllClasses(double[][] testInputs) {
        double[] predictions = new double[testInputs.length];

        for (int i = 0; i < predictions.length; i++) {
            double[] outputs = NNForwardPropagation(testInputs[i]);

            if (outputs.length == 1) {
                predictions[i] = outputs[0] >= 0.5 ? 0.0 : 1.0;
            } else {
                predictions[i] = MathsUtilities.IndexMaxOfArray(outputs);
            }
        }

        return predictions;
    }

    public static ILoss getLossFunction() {
        return lossFunction;
    }

    // ***************** HELPERS *****************

    public void DisplayPredictions(double[][] predictions) {
        for (int i = 0; i < predictions.length; i++) {
            System.out.println(i + "    " + Arrays.toString(predictions[i]));
        }
    }

    public void DisplayPredictionsWithColors(double[] predictions, double[] expectedOutputs) {
        for (int i = 0; i < predictions.length; i++) {

            if (predictions[i] == expectedOutputs[i]) {
                System.out.println("\u001B[32m" + "Prediction correct: " + predictions[i] + " == " + expectedOutputs[i] + "\u001B[0m");
            } else {
                System.out.println("\u001B[31m" + "Prediction incorrect: " + predictions[i] + " != " + expectedOutputs[i] + "\u001B[0m");
            }
        }
    }

    public void DisplayTestAccuracy(double[][] inputs, double[] expectedOutput) {
        double[][] expectedOutputs = OneHotEncoder(expectedOutput, inputs[0].length);
        double[][] predictions = OneHotEncoder(PredictAllClasses(inputs), inputs[0].length);
        System.out.println("Accuracy: " + GetCorrectPredictions(predictions, expectedOutputs) / (double) expectedOutput.length);
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
