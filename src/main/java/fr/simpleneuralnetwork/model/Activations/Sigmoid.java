package fr.simpleneuralnetwork.model.Activations;

import fr.simpleneuralnetwork.model.IActivation;

public class Sigmoid implements IActivation {

    private double Apply(double z) {
        return 1.0 / (1.0 + Math.exp(-1 * z));
    }

    public double Derivative(double z) {
        double sigmoid = Apply(z);
        return sigmoid * (1 - sigmoid);
    }

    public double[][] ApplyMatrix(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Apply(input[i][j]);
            }
        }
        return result;
    }

    @Override
    public double[][] DerivativeMatrix(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Derivative(input[i][j]);
            }
        }
        return result;
    }
}
