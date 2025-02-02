package fr.simpleneuralnetwork.model.Activations;

import fr.simpleneuralnetwork.model.IActivation;

import java.util.stream.IntStream;

public class ReLU implements IActivation {

    @Override
    public double Apply(double z) {
        return z > 0 ? z : 0;
    }

    @Override
    public double Derivative(double z) {
        return z > 0 ? 1 : 0;
    }
}
