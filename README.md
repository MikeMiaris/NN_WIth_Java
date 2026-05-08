
This elegant formulation arises from the derivative of the cross‑entropy loss with respect to the softmax inputs, which makes the implementation both correct and efficient.

### Backpropagation & Gradient Descent

The `Network` class implements **full backpropagation**:

1. **Output layer error:** For each output neuron, compute `delta = output - target` and update the bias.
2. **Hidden layer error propagation:** Starting from the last hidden layer and moving backward, deltas are computed using the chain rule:
   - The delta of each neuron in layer L is the **weighted sum of the deltas** of the next layer, multiplied by the **derivative of the activation function**.
   - This implements the standard backpropagation equations for multi‑layer networks (Rumelhart et al., 1986).

3. **Weight update:** After accumulating deltas across a mini‑batch, weights and biases are updated using the `learningRate` and the average gradient.

### Batch Training (Mini‑batch)

The training loop processes `N` training examples in **mini‑batches** of size `B` (where `B` divides `N`):

- When **B = 1**, gradients are updated after every single example (stochastic gradient descent – often noisier but faster).
- When **B = N**, gradients are accumulated over the entire training set (batch gradient descent – stable but computationally heavy).
- The actual code uses a **configurable** batch size.

The batch size is defined as `N/B` where `B` is set at the beginning of the program. The assignment explicitly requested testing with **B = N/20** and **B = N/200** to compare convergence speed and generalization.

### Weight Initialization

All weights and biases are initialized **randomly in the range (-1, 1)** at the start of training, as specified by the assignment. This ensures symmetry is broken and the network can learn properly.

---

## Implementation Details

### Key Components

| Class | Responsibility |
|-------|----------------|
| `Neuron.java` | Stores weights, bias, delta (gradient), and output; applies activation functions. |
| `Layer.java` | Groups multiple neurons; manages forward/backward operations for one layer. |
| `Network.java` | Coordinates layers; performs forward pass, softmax, backward pass, and weight updates. |
| `DataGenerator.java` | Generates synthetic training and test datasets as CSV files. |
| `DataLoader.java` | Reads CSV files and returns feature arrays and one‑hot encoded label arrays. |
| `NeuralNetworkMain.java` | Entry point: loads data, initializes the network, runs training, and reports results. |

### Training & Evaluation Loop

1. **Data Loading:** Load `classification_train.csv` and `classification_test.csv`.
2. **Network Initialization:** Create either a PT2 or PT3 `Network` with specified hyperparameters.
3. **Training:**
   - For a maximum number of epochs (default 1000), process mini‑batches.
   - After each epoch, calculate and print the **total training error**.
   - **Early stopping:** Terminate when the absolute difference in training error between two consecutive epochs falls below a `threshold` (e.g., 0.001), but only after at least **800 epochs** have completed (as required by the assignment).
4. **Evaluation:**
   - Compute and print the **generalization accuracy** (% correct predictions) on the test set.
   - Identify the best network configuration based on test accuracy.

---

## Experimental Setup & Hyperparameters

The assignment required systematic experimentation with:

- **Hidden layer sizes (H1, H2, H3):** multiple combinations tested.
- **Activation functions:** `tanh` vs `ReLU` in the deeper hidden layers.
- **Batch sizes:** `B = N/20` and `B = N/200`.
- **Learning rate:** default `0.001`.
- **Threshold:** `0.001` (for early stopping criterion).

The results were recorded in a table documented in the assignment report (PDF). The **best network** was then used to visualize test set predictions (correct vs misclassified) using distinct markers.

---

## How to Run

### Prerequisites
- Java Development Kit (JDK 8 or later)
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MikeMiaris/NN_WIth_Java.git
   cd NN_WIth_Java/NN_project

2. **Compile the source files:**
   javac DataGenerator.java DataLoader.java Layer.java Network.java NeuralNetworkMain.java

3. **Generate the datasets:**
   java DataGenerator

4. **Run the neural network:**
   java NeuralNetworkMain


## Note

The current main class loads a **PT3** network with:
- Hidden layers: `H1=3, H2=3, H3=3`
- Hidden activation: **ReLU**
- Output activation: **softmax**
- Learning rate: `0.001`
- Batch size: `20` (i.e., `N/20`)

You can modify these values directly in `NeuralNetworkMain.java`.

## Results & Observations

The network successfully learns to classify the four categories with high accuracy. The full experimental tables are included in the assignment report (`YN_project_2024-25.pdf`, in Greek).

Key observations:

- **PT3 (3 hidden layers)** generally achieves higher accuracy than **PT2 (2 hidden layers)** for this dataset, owing to its increased representational capacity.
- **tanh** tended to provide smoother convergence than **ReLU** for deeper networks, though ReLU avoided vanishing gradient issues when networks were deep.
- Smaller batch sizes (`B=N/200`) introduced more noise but often led to faster convergence in terms of epoch count.
- The classification accuracy approaches **~97‑99%** on the test set with well‑tuned hyperparameters.

For details on the second assignment (K‑means clustering) and the full report, please refer to the course materials.

## Future Work

- Implement **momentum** and **learning rate decay** to improve convergence.
- Add support for **dropout** regularization.
- Extend the network to handle **regression problems**.
- Implement the **K‑means clustering** part (Exercise 2) in the same Java framework.
- Provide a command‑line interface or configuration file to set hyperparameters.

## License

This project is for **educational/academic** purposes. Feel free to use and modify it. Original assignment credit: University of Ioannina, Department of Computer Science & Engineering, 2024‑2025.
