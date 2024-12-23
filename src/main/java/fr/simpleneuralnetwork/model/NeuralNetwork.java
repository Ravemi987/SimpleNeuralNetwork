package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.utils.MathsOperations;

import java.lang.reflect.Array;
import java.util.Arrays;

public class NeuralNetwork {

    public double F(double[] Xi, double[] Wi, double b) {
        return MathsOperations.Sigmoid(MathsOperations.Linear(Xi, Wi, b));
    }

    public double LocalError(double[] Xi, double[] Wi, double zi, double b) {
        double diff = (F(Xi, Wi, b) - zi);
        return diff * diff;
    }

    public double GlobalError(double[][] X, double[] W, double[] z, double[] b) {
        double E = 0;
        double N = X.length;

        for (int i = 0; i < N; i++) {
            E += LocalError(X[i], W, z[i], b[i]);
        }

        return (1.0 / N) * E;
    }

    public double[] LocalGradient(double[] Xi, double[] Wi, double zi, double b) {
        double[] grad = new double[Wi.length + 1];
        double linearOutput = MathsOperations.Linear(Xi, Wi, b);
        double sigmoidDerivative = MathsOperations.SigmoidDerivative(linearOutput);

        for (int i = 0; i < Wi.length; i++) {
            grad[i] = 2 * Xi[i] * sigmoidDerivative * (F(Xi, Wi, b) - zi);
        }
        grad[Wi.length] = 2 * sigmoidDerivative * (F(Xi, Wi, b) - zi);

        return grad;
    }

    public double[] GlobalGradient(double[][] X, double[] W, double[] z, double[] b) {
        int D = W.length + 1;
        double N = X.length;
        double[] global_grad = new double[D];

        for (int i = 0; i < N; i++) {
            double[] local_grad = LocalGradient(X[i], W, z[i], b[i]);
            for (int j = 0; j < D; j++) {
                global_grad[j] +=  local_grad[j];
            }
        }

        for (int j = 0; j < D; j++) {
            global_grad[j] /= N;
        }

        return global_grad;
    }

    public double[] UpdateWeights(double[][] X, double[] W, double[] z, double[] b, double learningRate) {
        int D = W.length + 1;
        double[] newWeights = Arrays.copyOf(W, W.length);
        double[] grad = GlobalGradient(X, W, z, b);

        for (int i = 0; i < D; i++) {
            newWeights[i] = W[i] - learningRate * grad[i];
        }

        b[0] -= learningRate * grad[W.length];

        return newWeights;
    }

    public static void main(String[] args) {
        double[][] X = {{-1, 2}, {3, 4}};
        double[] W = {0, 1};
        double b = -2;

        System.out.println(MathsOperations.Linear(X[0], W, b));
    }
}