package fr.simpleneuralnetwork.tests;

import fr.simpleneuralnetwork.model.NeuralNetwork;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class MNISTLoader {

    public static String path = "src/main/resources/MNIST/";

    public static int scan32BitsInteger(byte[] arr, int... indexes) {
        return ((arr[indexes[0]] & 0xFF) << 24) |
                ((arr[indexes[1]] & 0xFF) << 16) |
                ((arr[indexes[2]] & 0xFF) << 8) |
                (arr[indexes[3]] & 0xFF);
    }

    public static int[] readImagesHeader(String filePath) throws IOException {
        /*
        [offset] [type]          [value]          [description]
        0000     32-bit integer  0x00000803(2051) magic number
        0004     32-bit integer  60000            number of images
        0008     32-bit integer  28               number of rows
        0012     32-bit integer  28               number of columns
         */
        try (BufferedInputStream buffer = new BufferedInputStream(new FileInputStream(filePath))) {
            byte[] header = new byte[16];

            if (buffer.read(header) == -1) {
                return null;
            }

            int magicNumber = scan32BitsInteger(header, 0, 1, 2, 3);
            int numberOfImages = scan32BitsInteger(header, 4, 5, 6, 7);
            int numberOfRows = scan32BitsInteger(header, 8, 9, 10, 11);
            int numberOfColumns = scan32BitsInteger(header, 12, 13, 14, 15);

            return new int[]{magicNumber, numberOfImages, numberOfRows, numberOfColumns};
        }
    }

    public static int[] readLabelsHeader(String filePath) throws  IOException {
        /*
        [offset] [type]          [value]          [description]
        0000     32-bit integer  0x00000801(2049) magic number (MSB first)
        0004     32-bit integer  60000            number of items
         */
        try (BufferedInputStream buffer = new BufferedInputStream(new FileInputStream(filePath))) {
            byte[] header = new byte[8];

            if (buffer.read(header) == -1) {
                return null;
            }

            int magicNumber = scan32BitsInteger(header, 0, 1, 2, 3);
            int numberOfItems = scan32BitsInteger(header, 4, 5, 6, 7);

            return new int[]{magicNumber, numberOfItems};
        }
    }

    public static double[][] readImages(String filePath, int numberOfImages,
                                      int numberOfRows, int numberOfColumns) throws IOException {
        /*
        [offset] [type]          [value]          [description]
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
         */
        try (BufferedInputStream buffer = new BufferedInputStream(new FileInputStream(filePath))) {
            buffer.skip(16);
            int imageSize = numberOfRows * numberOfColumns;
            byte[][] images = new byte[numberOfImages][imageSize];

            for (int i = 0; i < numberOfImages; i++) {
                buffer.read(images[i]);
            }

            return NormalizeImages(images);
        }
    }

    public static double[] readLabels(String filePath, int numberOfExamples) throws IOException {
        /*
        [offset] [type]          [value]          [description]
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
         */
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filePath))) {
            bis.skip(8);

            double[] labels = new double[numberOfExamples];
            for (int i = 0; i < numberOfExamples; i++) {
                labels[i] = bis.read();
            }
            return labels;
        }
    }

    public static double[][] NormalizeImages(byte[][] images) {
        double[][] normalizedImages = new double[images.length][images[0].length];
        for (int i = 0; i < images.length; i++) {
            for (int j = 0; j < images[i].length; j++) {
                normalizedImages[i][j] = ((images[i][j] & 0xFF) / 255.0);
            }
        }
        return normalizedImages;
    }

    public static double[][] getTrainData() throws IOException {
        String filePath = path + "train-images.idx3-ubyte";
        int[] headerInfos = readImagesHeader(filePath);

        if (headerInfos != null)
            return readImages(filePath, headerInfos[1], headerInfos[2], headerInfos[3]);
        return null;
    }

    public static double[] getTrainLabels() throws IOException {
        String filePath = path + "train-labels.idx1-ubyte";
        int[] headerInfos = readLabelsHeader(filePath);

        if (headerInfos != null)
            return readLabels(filePath, headerInfos[1]);
        return null;
    }

    public static double[][] getTestData() throws IOException {
        String filePath = path + "t10k-images.idx3-ubyte";
        int[] headerInfos = readImagesHeader(filePath);

        if (headerInfos != null)
            return readImages(filePath, headerInfos[1], headerInfos[2], headerInfos[3]);
        return null;
    }

    public static double[] getTestLabels() throws IOException {
        String filePath = path + "t10k-labels.idx1-ubyte";
        int[] headerInfos = readLabelsHeader(filePath);

        if (headerInfos != null)
            return readLabels(filePath, headerInfos[1]);
        return null;
    }

    public static void DisplayImage(double[] image) {
        int imageSize = (int) Math.sqrt(image.length);

        for (int i = 0; i < imageSize; i++) {
            for (int j = 0; j < imageSize; j++) {
                double pixel = image[i * imageSize + j];
                System.out.print(pixel > 0 ? "# " : "." + " ");
            }
            System.out.print("\n");
        }
    }

    public static void main(String[] args) throws IOException {
        double[][] trainData = getTrainData();
        double[] trainLabels = getTrainLabels();
        double[][] testData = getTestData();
        double[] testLabels = getTestLabels();

        if (trainData == null || trainLabels == null || testData == null || testLabels == null) {
            return;
        }

        System.out.println("trainData: length=" + trainData.length + " features=" + trainData[0].length);
        System.out.println("testData: length=" + testData.length + " features=" + testData[0].length);
        System.out.println("trainLabels: length=" + trainLabels.length);
        System.out.println("testLabels: length=" + testLabels.length);

        int[] layerSizes = new int[]{784, 128, 64, 10};

        NeuralNetwork nn = new NeuralNetwork(layerSizes, "CrossEntropy", "Relu", "Softmax");
        nn.Train(trainData, trainLabels, 0.01, 10, 100, 1E-7);

        double[] predictions = nn.PredictAllClasses(testData);

//        for (int i = 0; i < predictions.length; i++) {
//            if (predictions[i] != testLabels[i]) {
//                DisplayImage(testData[i]);
//                System.out.println("\u001B[31m" + "Prediction incorrect: " + predictions[i] + ". Attendue: " + testLabels[i] + "\u001B[0m");
//                System.out.println("-------------------------------------------------------");
//            }
//        }
        nn.DisplayTestAccuracy(testData, testLabels);
    }
}
