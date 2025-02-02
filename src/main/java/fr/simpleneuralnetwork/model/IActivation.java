package fr.simpleneuralnetwork.model;

import java.util.stream.IntStream;

public interface IActivation {

    double Apply(double z);
    double Derivative(double z);

    default double[][] ApplyMatrix(double[][] input) {
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

    default double[][] DerivativeMatrix(double[][] input) {
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
