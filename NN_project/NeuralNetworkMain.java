
import java.io.IOException;



public class NeuralNetworkMain {

	public static void main(String[] args) {

		//ADD: USER INPUT FOR THE HYPERPARAMETERS
		int d = 2;
		int H1 = 3;
		int H2 = 3;
		int H3 = 3;
		int categories = 4;
		int hiddenFunction = 3;  //ACTIVATION FUNCTIONS: 0:NO ACTIVATION 1:SIGMOID, 2:TANH, 3:RELU
		int outputFunction = 1;
		double learningRate = 0.001;
		double threshold = 0.001;
		int epochs = 1000;
		int batchSize = 20;
		
		
		//Loading Training Set
		double[][] trainInputs = new double[0][0];
		double[][] trainOutputs = new double[0][0];
		
        try {
            String[] classificationCategories = {"C1", "C2", "C3", "C4"};

            trainInputs = DataLoader.loadInputs("classification_train.csv");
            trainOutputs = DataLoader.loadOutputs("classification_train.csv", classificationCategories);

            System.out.println("Train Inputs: " + trainInputs.length + " παραδείγματα");
            System.out.println("Train Outputs: " + trainOutputs.length + " παραδείγματα");

            System.out.println("\nΔείγμα Εισόδων:");
            for (int i = 0; i < Math.min(5, trainInputs.length); i++) {
                System.out.println("Παράδειγμα " + (i + 1) + ": " + 
                    java.util.Arrays.toString(trainInputs[i]));
            }

            System.out.println("\nΔείγμα Εξόδων:");
            for (int i = 0; i < Math.min(5, trainOutputs.length); i++) {
                System.out.println("Παράδειγμα " + (i + 1) + ": " + 
                    java.util.Arrays.toString(trainOutputs[i]));
            }
            checkInputRanges(trainInputs);
            countClasses(trainOutputs, classificationCategories);
            verifyData(trainInputs, trainOutputs, classificationCategories);

        } catch (IOException e) {
            System.err.println("Σφάλμα κατά τη φόρτωση δεδομένων: " + e.getMessage());
        }
		
        
        //Loading Test Set
		double[][] testInputs = new double[0][0];
		double[][] testOutputs = new double[0][0];
       
        try {
            String[] classificationCategories = {"C1", "C2", "C3", "C4"};

            testInputs = DataLoader.loadInputs("classification_test.csv");
            testOutputs = DataLoader.loadOutputs("classification_test.csv", classificationCategories);

            System.out.println("Test Inputs: " + testInputs.length + " παραδείγματα");
            System.out.println("Test Outputs: " + testOutputs.length + " παραδείγματα");

            System.out.println("\nΔείγμα Εισόδων:");
            for (int i = 0; i < Math.min(5, testInputs.length); i++) {
                System.out.println("Παράδειγμα " + (i + 1) + ": " + 
                    java.util.Arrays.toString(testInputs[i]));
            }

            System.out.println("\nΔείγμα Εξόδων:");
            for (int i = 0; i < Math.min(5, testOutputs.length); i++) {
                System.out.println("Παράδειγμα " + (i + 1) + ": " + 
                    java.util.Arrays.toString(testOutputs[i]));
            }
            checkInputRanges(testInputs);
            countClasses(testOutputs, classificationCategories);
            verifyData(testInputs, testOutputs, classificationCategories);

        } catch (IOException e) {
            System.err.println("Σφάλμα κατά τη φόρτωση δεδομένων: " + e.getMessage());
        }
        
        Network network1 = new Network(d, H1, H2, H3, categories , hiddenFunction, outputFunction);
		
		//network1.printNetwork();
		network1.train(trainInputs, trainOutputs, epochs, batchSize, learningRate, threshold);
		//network1.printNetwork();
		network1.test(testInputs,testOutputs);

	}

	
	public static void checkInputRanges(double[][] inputs) {
        boolean isValid = true;
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[i].length; j++) {
                if (inputs[i][j] < -1 || inputs[i][j] > 1) {
                    isValid = false;
                    System.out.println("Σφάλμα στο παράδειγμα " + (i + 1) + ", τιμή: " + inputs[i][j]);
                }
            }
        }
        if (isValid) {
            System.out.println("Όλες οι τιμές εισόδου είναι εντός του εύρους [-1, 1]");
        }
    }

    public static void countClasses(double[][] outputs, String[] categories) {
        int[] classCounts = new int[categories.length];
        for (double[] output : outputs) {
            for (int i = 0; i < output.length; i++) {
                if (output[i] == 1.0) {
                    classCounts[i]++;
                    break;
                }
            }
        }
        for (int i = 0; i < categories.length; i++) {
            System.out.println(categories[i] + ": " + classCounts[i] + " παραδείγματα");
        }
    }

    public static void verifyData(double[][] inputs, double[][] outputs, String[] categories) {
        boolean isValid = true;
        for (int i = 0; i < inputs.length; i++) {
            String expectedLabel = DataGenerator.classifyPoint(inputs[i][0], inputs[i][1]);
            int expectedIndex = java.util.Arrays.asList(categories).indexOf(expectedLabel);
            for (int j = 0; j < outputs[i].length; j++) {
                if ((j == expectedIndex && outputs[i][j] != 1.0) || (j != expectedIndex && outputs[i][j] != 0.0)) {
                    isValid = false;
                    System.out.println("Σφάλμα στο παράδειγμα " + (i + 1));
                    System.out.println("Είσοδος: " + java.util.Arrays.toString(inputs[i]));
                    System.out.println("Πραγματική έξοδος: " + java.util.Arrays.toString(outputs[i]));
                    System.out.println("Αναμενόμενη κατηγορία: " + expectedLabel);
                    break;
                }
            }
            if (!isValid) break;
        }
        if (isValid) {
            System.out.println("Όλα τα δεδομένα αντιστοιχούν σωστά!");
        }
    }

    
}
