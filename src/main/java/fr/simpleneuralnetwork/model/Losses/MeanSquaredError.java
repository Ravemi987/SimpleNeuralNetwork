package fr.simpleneuralnetwork.model.Losses;

import fr.simpleneuralnetwork.model.ILoss;

public class MeanSquaredError implements ILoss {

    public double Apply(double output, double expectedOutput) {
        double diff = output - expectedOutput;
        return diff * diff;
    }

    public double Derivative(double output, double expectedOutput) {
        return 2 * (output - expectedOutput);
    }

    @Override
    public double[][] DerivativeMatrix(double[][] output, double[][] expectedOutput) {
        int rows = output.length;
        int cols = output[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Derivative(output[i][j], expectedOutput[i][j]);
            }
        }
        return result;
    }
}
