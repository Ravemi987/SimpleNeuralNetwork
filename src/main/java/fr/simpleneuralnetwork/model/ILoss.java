package fr.simpleneuralnetwork.model;

public interface ILoss {

    double Apply(double output, double expectedOutput);
    double Derivative(double output, double expectedOutput);
    double Loss(double[] output, double[] expectedOutputs);
    double GlobalLoss(double[][] outputs, double[][] expectedOutputs);

    default double[][] DerivativeMatrix(double[][] output, double[][] expectedOutput) {
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
