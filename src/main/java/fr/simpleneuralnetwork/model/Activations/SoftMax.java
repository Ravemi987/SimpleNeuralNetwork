package fr.simpleneuralnetwork.model.Activations;

import fr.simpleneuralnetwork.model.IActivation;

public class SoftMax implements IActivation {

    private double[] Apply(double[] z) {
        double sum = 0;
        double[] res = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            res[i] = Math.exp(z[i]);
            sum += res[i];
        }

        for (int i = 0; i < z.length; i++) {
            res[i] /= sum;
        }

        return res;
    }

    private double[] Derivative(double[] z) {
        double[] softmax = Apply(z);
        double[] res = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            res[i] = softmax[i] * (1 - softmax[i]);
        }

        return res;
    }

    @Override
    public double[][] ApplyMatrix(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            result[i] = Apply(input[i]);
        }

        return result;
    }

    @Override
    public double[][] DerivativeMatrix(double[][] input) {
        int rows = input.length;
        int cols = input[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            result[i] = Derivative(input[i]);
        }

        return  result;
    }
}
