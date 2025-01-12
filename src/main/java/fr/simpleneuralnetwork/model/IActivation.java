package fr.simpleneuralnetwork.model;

public interface IActivation {

    double[][] ApplyMatrix(double[][] input);
    double[][] DerivativeMatrix(double[][] input);
}
