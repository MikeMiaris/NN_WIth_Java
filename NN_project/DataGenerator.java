
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class DataGenerator {
    
	public static void main(String[] args) throws IOException {
        int numTrain = 4000; //training data
        int numTest = 4000;  //test data
        String trainFile = "classification_train.csv";
        String testFile = "classification_test.csv";

        // data (train + test)
        generateClassificationData(numTrain, trainFile);
        generateClassificationData(numTest, testFile);
    }

    public static void generateClassificationData(int numPoints, String fileName) throws IOException {
        Random random = new Random();
        FileWriter writer = new FileWriter(fileName);
        writer.write("x1,x2,label\n");

        for (int i = 0; i < numPoints; i++) {
            double x1 = -1 + 2 * random.nextDouble(); //random x1 in: [-1, 1]
            double x2 = -1 + 2 * random.nextDouble(); //random x2 in: [-1, 1]
            String label = classifyPoint(x1, x2);
            writer.write(x1 + "," + x2 + "," + label + "\n");
        }

        writer.close();
        System.out.println("Data saved to " + fileName);
        //
        System.out.println("Τα αρχεία αποθηκεύτηκαν στο: " + new java.io.File(".").getAbsolutePath());

    }

    public static String classifyPoint(double x1, double x2) {
        if (Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2 && x2 > 0.5) {
            return "C1";
        } else if (Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2 && x2 < 0.5) {
            return "C2";
        } else if (Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2 && x2 > -0.5) {
            return "C1";
        } else if (Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2 && x2 < -0.5) {
            return "C2";
        } else if (Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2 && x2 > -0.5) {
            return "C1";
        } else if (Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2) < 0.2 && x2 < -0.5) {
            return "C2";
        } else if (Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2 && x2 > 0.5) {
            return "C1";
        } else if (Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2) < 0.2 && x2 < 0.5) {
            return "C2";
        } else if (x1 * x2 > 0) {
            return "C3";
        } else {
            return "C4";
        }
    }

}
