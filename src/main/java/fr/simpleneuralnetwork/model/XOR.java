package fr.simpleneuralnetwork.model;

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

        // ********************* RESULTAT ***********************
        double[][] weightsFinalLayer1 = {
                {3.562353040620003, 3.567595413450272},
                {-5.368272647651077, -5.393533380252852}
        };
        double[] biasesFinalLayer1 = { -5.368272647651077, 1.8403347560151768};

        double[][] weightsFinalLayer2 = {
                {-6.323484381327753, -6.380110986024997},
        };
        double[] biasesFinalLayer2 = {3.1216910185491393};

        double[][][] finalWeights = {weightsFinalLayer1, weightsFinalLayer2};
        double[][] finalBiases = {biasesFinalLayer1, biasesFinalLayer2};
        // *************************************************************

        // ********************* POIDS INITIAUX ***********************
        double[][] weightsLayer1 = {
                {1.0, 2.0},
                {-3.0, -2.0}
        };
        double[] biasesLayer1 = {-3.0, 1.0};

        double[][] weightsLayer2 = {
                {1.0, 1.0},
        };
        double[] biasesLayer2 = {-1.0};

        double[][][] initialWeights = {weightsLayer1, weightsLayer2};
        double[][] initialBiases = {biasesLayer1, biasesLayer2};
        // *************************************************************

        int[] layerSizes = {2, 2, 1}; // Deux entrées, une couche cachée avec deux neurones, un neurone de sortie
        double[][] trainInputs = generateTrainInputs();
        double[] trainOutputs = generateOutputs();

        NeuralNetwork nn = new NeuralNetwork(initialWeights, initialBiases, layerSizes);

        System.out.println("Training...");
        nn.Train(trainInputs, trainOutputs, 1, 1000);

        System.out.println("Predicting output...");
        System.out.println(Arrays.toString(nn.GetAllWeights()));
        System.out.println(Arrays.toString(nn.Predict(trainInputs)));
    }
}
