package fr.simpleneuralnetwork.model.Activations;

import fr.simpleneuralnetwork.model.IActivation;

import java.util.stream.IntStream;

public class ReLU implements IActivation {
    private final double alpha = 0.01;

    @Override
    public double Apply(double z) {
        return z > 0 ? z : alpha * z;
    }

    @Override
    public double Derivative(double z) {
        return z > 0 ? 1 : alpha;
    }
}
