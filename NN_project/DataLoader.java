
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    
	public static double[][] loadInputs(String fileName) throws IOException {
        List<double[]> inputsList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line = br.readLine(); //skip header
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double x1 = Double.parseDouble(parts[0]);
                double x2 = Double.parseDouble(parts[1]);
                inputsList.add(new double[] {x1, x2});
            }
        }
        return inputsList.toArray(new double[0][]);
    }

    public static double[][] loadOutputs(String fileName, String[] categories) throws IOException {
        List<double[]> outputsList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line = br.readLine();
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                String label = parts[2];
                outputsList.add(oneHotEncode(label, categories));
            }
        }
        return outputsList.toArray(new double[0][]);
    }

    private static double[] oneHotEncode(String label, String[] categories) {
        double[] encoding = new double[categories.length];
        for (int i = 0; i < categories.length; i++) {
            if (categories[i].equals(label)) {
                encoding[i] = 1.0;
                break;
            }
        }
        return encoding;
    }

}