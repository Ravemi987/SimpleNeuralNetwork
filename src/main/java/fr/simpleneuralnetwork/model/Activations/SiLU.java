package fr.simpleneuralnetwork.model.Activations;

import fr.simpleneuralnetwork.model.IActivation;

public class SiLU implements IActivation {

    @Override
    public double Apply(double z) {
        return z / (1.0 + Math.exp(-z));
    }

    @Override
    public double Derivative(double z) {
        double exp = Math.exp(-z);
        return (1.0 + exp + z * exp) / ((1 + exp)*(1 + exp));
    }
}
