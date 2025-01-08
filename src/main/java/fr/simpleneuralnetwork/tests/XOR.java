package fr.simpleneuralnetwork.tests;

import fr.simpleneuralnetwork.model.NeuralNetwork;

import java.util.Arrays;

public class XOR {
    public static double[][] generateTrainInputs() {
        return new double[][] {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
    }

    public static double[] generateOutputs() {
        return new double[] {0, 1, 1, 0};
    }

    public static void main(String[] args) {
        /*
         * Reseau de neurones à deux couches, avec respectivements deux perceptrons,
         * et un perceptron. Fonction d'activation sigmoïde.
         * Poids initiaux = (1.0, 2.0, −3.0, −3.0, −2.0, 1.0, 1.0, 1.0, −1.0)
         */

        // ********************* POIDS INITIAUX ***********************
        double[][] weightsLayer1 = {
                {1.0, 2.0},
                {-3.0, -2.0}
        };
        double[] biasesLayer1 = {-3.0, 1.0};

        double[][] weightsLayer2 = {
                {1.0, 1.0}
        };
        double[] biasesLayer2 = {-1.0};

        double[][][] initialWeights = {weightsLayer1, weightsLayer2};
        double[][] initialBiases = {biasesLayer1, biasesLayer2};
        // *************************************************************

        int[] layerSizes = {2, 2, 1}; // Deux entrées, une couche cachée avec deux neurones, un neurone de sortie
        double[][] trainInputs = generateTrainInputs();
        double[] trainOutputs = generateOutputs();

        NeuralNetwork nn = new NeuralNetwork(initialWeights, initialBiases, layerSizes);
        NeuralNetwork nn2 = new NeuralNetwork(layerSizes);

        System.out.println("Training...");
        nn.Train(trainInputs, trainOutputs, 1, 1000, 4, 1.0E-4);

        System.out.println("Predicting output...");
        System.out.println(Arrays.toString(nn.GetAllWeights()));
        double[][] pred = nn.PredictAll(trainInputs);
        nn.DisplayPredictions(pred);

        System.out.println(Arrays.toString(nn.PredictAllClasses(trainInputs)));
        nn.DisplayTestAccuracy(trainInputs, trainOutputs);
    }
}
