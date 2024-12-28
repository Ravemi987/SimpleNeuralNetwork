package fr.simpleneuralnetwork.model;

import fr.simpleneuralnetwork.utils.MathsOperations;

import java.util.Random;

public class NeuralNetwork {
    private double[][] Inputs;
    private double[] Weights;
    private double[] Output;
    private double LearningRate;
    private double IterationsNumber;

    Random rand = new Random();

    public NeuralNetwork(double[][] Inputs, double[] Output, double LearningRate, double IterationsNumber) {
        this.Inputs = Inputs;
        this.Output = Output;
        this.LearningRate = LearningRate;
        this.IterationsNumber = IterationsNumber;
    }

    public double F(double[] X, double[] W) {
        return MathsOperations.Sigmoid(MathsOperations.Linear(X, W));
    }

    public double LocalError(double[] X, double[] W, double z) {
        double diff = (F(X, W) - z);
        return diff * diff;
    }

    public double GlobalError() {
        double E = 0;
        double N = Inputs.length;

        for (int i = 0; i < N; i++) {
            E += LocalError(Inputs[i], Weights, Output[i]);
        }

        return (1.0 / N) * E;
    }

    public double[] LocalGradient(double[] X, double[] W, double z) {
        double[] grad = new double[W.length];
        double linearOutput = MathsOperations.Linear(X, W);
        double sigmoidDerivative = MathsOperations.SigmoidDerivative(linearOutput);

        for (int i = 0; i < W.length; i++) {
            grad[i] = 2 * X[i] * sigmoidDerivative * (F(X, W) - z);
        }

        return grad;
    }

    public double[] GlobalGradient() {
        int N = Inputs.length;
        int D = Weights.length;
        double[] global_grad = new double[D];

        for (int i = 0; i < N; i++) {
            double[] local_grad = LocalGradient(Inputs[i], Weights, Output[i]);
            for (int j = 0; j < D; j++) {
                global_grad[j] +=  local_grad[j];
            }
        }

        for (int j = 0; j < D; j++) {
            global_grad[j] /= N;
        }

        return global_grad;
    }

    public void UpdateWeights() {
        int D = Weights.length;
        double[] grad = GlobalGradient();

        for (int i = 0; i < D; i++) {
            Weights[i] = Weights[i] - LearningRate * grad[i];
        }
    }

    public double[][] ReshapeInput(double[][] Inputs) {
        int rows = Inputs.length;
        int cols = Inputs[0].length;
        double[][] InputWithBias = new double[rows][cols + 1];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(Inputs[i], 0, InputWithBias[i], 0, cols);
            InputWithBias[i][cols] = 1.0;
        }

        return InputWithBias;
    }

    public void InitWeights() {
        int N = Inputs[0].length;
        Weights = new double[N];

        for (int i  = 0; i < N; i++) {
            Weights[i] = rand.nextDouble();
        }
    }

    public void GradientDescent() {
        Inputs = ReshapeInput(Inputs);
        InitWeights();

        for (int k = 0; k < IterationsNumber; k++) {
            UpdateWeights();
        }
    }

    public double[] Train() {
        GradientDescent();
        return Weights;
    }

    public double[] Predict(double[][] TestInputs) {
        int rows = TestInputs.length;
        double[] pred = new double[rows];

        double[][] inputsWithBias = ReshapeInput(Inputs);

        for (int i = 0; i < rows; i++) {
            pred[i] = F(inputsWithBias[i], Weights);
        }

        return pred;
    }

    public static void main(String[] args) {
        System.out.println("No execution error");
    }
}