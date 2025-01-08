package fr.simpleneuralnetwork.utils;

import java.util.function.Function;

public class MathsUtilities {

    // Old method (too slow)
    public static double Linear(double[] X, double[] W, double b) {
        double r = 0;
        int len = X.length;

        for (int i = 0; i < len; ++i) {
            r += X[i] * W[i];
        }
        return r + b;
    }

    public static double Sigmoid(double z) {
        return 1 / (1 + Math.exp(-1 * z));
    }

    public static double SigmoidDerivative(double z) {
        double sigmoid = Sigmoid(z);
        return sigmoid * (1 - sigmoid);
    }

    public static int IndexMaxOfArray(double[] arr) {
        int maxIndex = 0;

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static double[][] ApplyActivation(double[][] Z, Function<Double, Double> function) {
        int rows = Z.length;
        int cols = Z[0].length;
        double[][] A = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                A[i][j] = function.apply(Z[i][j]);
            }
        }
        return A;
    }

}
