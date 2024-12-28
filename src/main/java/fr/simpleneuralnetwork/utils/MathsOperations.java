package fr.simpleneuralnetwork.utils;

public class MathsOperations {

    public static double Linear(double[] Xi, double[] Wi) {
        double r = 0;

        for (int i = 0; i < Xi.length; i++) {
            r += Xi[i] * Wi[i];
        }
        return r;
    }

    public static double Sigmoid(double z) {
        return 1 / (1 + Math.exp(-1 * z));
    }

    public static double SigmoidDerivative(double z) {
        double sigmoid = Sigmoid(z);
        return sigmoid * (1 - sigmoid);
    }
}
