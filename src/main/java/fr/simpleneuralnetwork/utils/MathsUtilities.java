package fr.simpleneuralnetwork.utils;

public class MathsUtilities {

    public static double Linear(double[] X, double[] W, double b) {
        double r = 0;
        int len = X.length; // Optimisation

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
}
