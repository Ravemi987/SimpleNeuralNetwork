package fr.simpleneuralnetwork.model;

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
}
