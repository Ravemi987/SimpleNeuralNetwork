package fr.simpleneuralnetwork.model;

public interface ILoss {

    double Apply(double output, double expectedOutput);
    double[][] DerivativeMatrix(double[][] output, double[][] expectedOutput);
}
