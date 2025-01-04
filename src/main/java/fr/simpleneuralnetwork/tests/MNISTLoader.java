package fr.simpleneuralnetwork.tests;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTLoader {

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

    public static byte[][] readImages(String filePath, int numberOfImages,
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
            return images;
        }
    }

    public static byte[] readLabels(String filePath, int numberOfExamples) throws IOException {
        /*
        [offset] [type]          [value]          [description]
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
         */
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(filePath))) {
            bis.skip(8);

            byte[] labels = new byte[numberOfExamples];
            for (int i = 0; i < numberOfExamples; i++) {
                labels[i] = (byte) bis.read();
            }
            return labels;
        }
    }

    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}