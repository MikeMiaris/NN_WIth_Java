
import java.util.ArrayList;
import java.util.List;

public class Network {
	
	List<Layer> layers = new ArrayList<>();
	
	//Constructor for PT2
	public Network(int d, int H1, int H2 , int categories , int hiddenFunction, int outputFunction) {

		//input layer
		Layer inputLayer = new Layer(d, 0, 1);
		this.layers.add(inputLayer);
		
		//hidden layers
		Layer layer1 = new Layer(H1, 2, d);
		this.layers.add(layer1);
		
		Layer layer2 = new Layer(H2, hiddenFunction, H1);
		this.layers.add(layer2);
		
		//output layer
		Layer layerOut = new Layer(categories, outputFunction, H2);
		this.layers.add(layerOut);
	}
	
	//Constructor for PT3
	public Network(int d, int H1, int H2 ,int H3, int categories , int hiddenFunction, int outputFunction) {

		//input layer
		Layer inputLayer = new Layer(d, 0, 1);
		this.layers.add(inputLayer);
		
		//hidden layers
		Layer layer1 = new Layer(H1, 2, d);
		this.layers.add(layer1);
		
		Layer layer2 = new Layer(H2, 2, H1);
		this.layers.add(layer2);
		
		Layer layer3 = new Layer(H3, hiddenFunction, H2);
		this.layers.add(layer3);
		
		//output layer
		Layer layerOut = new Layer(categories, outputFunction, H3);
		this.layers.add(layerOut);
	}
	
	//Getter for Layers
	public List<Layer> getLayers() {
		return this.layers;
	}
	
	//Loading Values to Input Layer
	public void loadInputs(double [][] inputs,int index) {
		
		Layer inputLayer = this.layers.get(0);
		
		for(int i = 0; i<inputLayer.getSize(); i++) {
			this.layers.get(0).getNeurons().get(i).addInput(inputs[index][i]);
		}
	}
	
	//Forward-Pass
	public void forward() {
		for (int i =0; i<this.layers.size()-1; i++) {
			
			
			for(int j = 0; j < this.layers.get(i+1).getSize(); j++) {
				
				for(int k = 0; k < this.layers.get(i).getSize(); k++) {
					
					this.layers.get(i).getNeurons().get(k).calculate();
					
					double out = this.layers.get(i).getNeurons().get(k).getOutput();
					
					this.layers.get(i+1).getNeurons().get(j).addInput(out);	
				}
			}
			
		}
	}
	
	//Outputs Calculation for the Output Layer
	public void output() {	
	    Layer outputLayer = this.layers.get(this.layers.size() - 1);
	    for (Neuron neuron : outputLayer.getNeurons()) {
	    	double sum = 0.0;
	        for (int i = 0; i < neuron.getInputs().size(); i++) {
	            sum += neuron.getInputs().get(i) * neuron.getWeights(i);
	        }
	        sum += neuron.getBias();
	        neuron.setOutput(sum);
	    }
	    Softmax(outputLayer);
		
	}
	
	//Cross-Entropy Loss
	public double crossEntropyLoss(double[][] target, int index) { 
	    int lastLayerIndex = this.layers.size() - 1;
	    Layer outputLayer = this.layers.get(lastLayerIndex);

	    double sumLoss = 0.0;

	    for (int i = 0; i < outputLayer.getSize(); i++) {
	        double targetValue = target[index][i];
	        double outputValue = outputLayer.getNeurons().get(i).getOutput();

	        outputValue = Math.max(outputValue, 1e-9);
	        outputValue = Math.min(outputValue, 1 - 1e-9);

	        sumLoss += -targetValue * Math.log(outputValue);
	    }

	    return sumLoss / outputLayer.getSize();
	}

	//Softmax
	public void Softmax(Layer outputLayer) {
	    List<Neuron> neurons = outputLayer.getNeurons();
	    double min = Double.NEGATIVE_INFINITY;
	    
	    for (Neuron neuron : neurons) {
	        min = Math.max(min, neuron.getOutput());
	    }
	    
	    double sumExp = 0.0;
	    
	    for (Neuron neuron : neurons) {
	        sumExp += Math.exp(neuron.getOutput()-min);
	    }

	    for (Neuron neuron : neurons) {
	        double softmaxOutput = Math.exp(neuron.getOutput()-min)/sumExp;
	        neuron.setOutput(softmaxOutput);
	    }

	}
	
	//Backpropagation
	public void backprop(double[][] target,int index, double learningRate) {
		
		int lastLayerIndex = this.layers.size() - 1;
		Layer outputLayer = this.layers.get(lastLayerIndex);
		
		//compute Output errors(deltas)
		for (int i = 0; i < outputLayer.getSize(); i++) {
			Neuron neuron = outputLayer.getNeurons().get(i);
			double output = neuron.getOutput();
			
			double targetValue = target[index][i];
			
			//We use softmax so the delta is this
			double newDelta = output - targetValue;
			neuron.setDelta(newDelta);

			double newBias = neuron.getBias() - learningRate * neuron.getDelta();
			neuron.setBias(newBias);
		}
		
		
		
		//Compute hidden layer deltas
		for(int i = lastLayerIndex-1; i > 0 ; i--) {
			Layer frontLayer = this.layers.get(i+1);
			Layer backLayer = this.layers.get(i);
			
			List<Neuron> frontNeurons = frontLayer.getNeurons();
			
			double[] frontDeltas = new double[frontLayer.getSize()];
			double[][] weights = new double[backLayer.getSize()][frontLayer.getSize()];
			
			//fill deltas list
			for(int j = 0; j < frontLayer.getSize(); j++) {
				Neuron neuron = frontNeurons.get(j);
				double delta = neuron.getDelta();
				frontDeltas[j] = delta;
				
			}
			
			//fill weights list
			for(int j = 0; j < backLayer.getSize(); j++) { //gia thn diplh lista weights
				for(int k = 0; k < frontLayer.getSize(); k++) {
					Neuron frontNeuron = frontLayer.getNeurons().get(k);
					weights[j][k] = frontNeuron.getWeights(j);
				}	
			}
			
			//do the math
			double sum = 0;
			for(int j = 0; j < backLayer.getSize(); j++) { //gia thn diplh lista weights
				
				Neuron backNeuron = backLayer.getNeurons().get(j);
				
				for(int k = 0; k < frontLayer.getSize(); k++) {
					sum += weights[j][k] * frontDeltas[k];
				}
				
				double deriv = backNeuron.getDerivative();
				double newDelta = deriv * sum;
				
				
				backNeuron.setDelta(newDelta);
				
				
				sum = 0;
			}
			
		}
		
		
		//Weights update
		for (int i = 1; i <= lastLayerIndex; i++) {
		    Layer currentLayer = this.layers.get(i);
		    Layer previousLayer = this.layers.get(i - 1);
		    for (Neuron currentNeuron : currentLayer.getNeurons()) {
		        for (int j = 0; j < previousLayer.getSize(); j++) {
		            double input = previousLayer.getNeurons().get(j).getOutput();
		            double newWeight = currentNeuron.getWeights(j) - learningRate * currentNeuron.getDelta() * input;
		            currentNeuron.getWeights().set(j, newWeight);
		        }
		        double newBias = currentNeuron.getBias() - learningRate * currentNeuron.getDelta();
		        currentNeuron.setBias(newBias);
		    }
		}

		//clear old inputs now that backdrop is done
		for (int j = 0; j < this.layers.size(); j++) {
	        Layer layer = this.layers.get(j);

	        for (Neuron neuron : layer.getNeurons()) {
	            neuron.getInputs().clear();
	        }
	    }	
		
	}

	//Training Loop
	public void train(double[][] trainInputs, double[][] trainOutputs, int epochs, int batchSize, double learningRate, double threshold) {
	    
		int numExamples = trainInputs.length;
	    int numBatches = numExamples / batchSize;

	    double prevError = Double.MAX_VALUE;
	    for (int epoch = 1; epoch <= epochs; epoch++) {
	        
	    	double totalError = 0.0;

	        //dynamic learning rate a good policy
	        if (epoch % 100 == 0) {
	            learningRate *= 0.9;
	            System.out.println("Epoch " + epoch + ": Learning rate adjusted to " + learningRate);
	        }     
	        
	        //also a good policy
	        shuffleData(trainInputs, trainOutputs);
	        
	        for (int batch = 0; batch < numBatches; batch++) {
	            for (int i = 0; i < batchSize; i++) {
	                
	            	int index = batch * batchSize + i;
	                
	                loadInputs(trainInputs, index);
	                forward();
	                output();
	                
	                totalError += crossEntropyLoss(trainOutputs, index);
	                
	                backprop(trainOutputs,index, learningRate);
	                
	            }
	        }
	        

	        double avgError = totalError / numExamples;
	        System.out.println("Epoch "+epoch+": Average Error = " + avgError);

	        if (epoch >= 800 && Math.abs(prevError - avgError) < threshold) {
	            System.out.println("Training converged at epoch " + epoch);
	            break;
	        }

	        prevError = avgError;
	    }

	    System.out.println("Training completed.");
	}
	
	//Network Accuracy Tester
	public void test(double[][] inputs,double[][] targets) {
		
		int correct = 0;
		
		for(int i = 0; i < inputs.length; i++) {
	        System.out.println("Input values for test example " + i + ":");
	        for (int k = 0; k < inputs[i].length; k++) {
	            System.out.println(inputs[i][k] + " ");
	        }
			loadInputs(inputs,i);
			forward();
			output();
			
	        int lastLayerIndex1 = this.layers.size() - 1;
	        Layer outputLayer1 = this.layers.get(lastLayerIndex1);
	        for (int j = 0; j < outputLayer1.getSize(); j++) {
	            Neuron neuron = outputLayer1.getNeurons().get(j);
	            System.out.println("Softmax output for class " + j + ": " + neuron.getOutput());
	        }
			
			//clear inputs
			for (int j = 0; j < this.layers.size(); j++) {
		        Layer layer = this.layers.get(j);

		        for (Neuron neuron : layer.getNeurons()) {
		            neuron.getInputs().clear();
		        }
		    }	
			
			
			int lastLayerIndex = this.layers.size() - 1;
			Layer outputLayer = this.layers.get(lastLayerIndex);
			
			int actualClass = 0;
			
			
			for(int j = 0; j<targets[i].length; j++ ) {
				if(Math.abs(targets[i][j] - 1.0) < 1e-9) {
					actualClass = j;
				}
			}
			System.out.println("Actual Class = "+actualClass);
			
			double maxOutput = Double.NEGATIVE_INFINITY;
			int prediction = 0;
			for (int j = 0; j < outputLayer.getSize(); j++) {
			    Neuron neuron = outputLayer.getNeurons().get(j);
			    double output = neuron.getOutput();
			    if (output > maxOutput) {
			        maxOutput = output;
			        prediction = j;
			    }
			}
			System.out.println("Prediction = "+prediction);

			
			if(actualClass == prediction) {
				correct ++;
				//System.out.println(correct);
			}			
		}
		double percentage = (double)correct/(double)inputs.length;
		System.out.println("Accuracy of model = " + percentage * 100+"%");

	}
	
	//just for checking things
	public void printNetwork() {
		
		for(int i=0; i < this.layers.size(); i++) {
			System.out.println("\nLayer"+i);
			Layer currentLayer = this.layers.get(i);
			
			for(int j = 0; j< currentLayer.getSize(); j++) {
				System.out.println("Neuron"+j);
				List<Double> weights = currentLayer.getNeurons().get(j).getWeights();
				System.out.println("Incoming Weights:"+weights);
				
				List<Double> inputs = currentLayer.getNeurons().get(j).getInputs();
				System.out.println("Inputs:"+inputs);
				
				double bias = currentLayer.getNeurons().get(j).getBias();
				System.out.println("Bias: " + bias);
				
				System.out.println("Output:"+currentLayer.getNeurons().get(j).getOutput());
			
			}
		}
	}
	
	//used in each epoch to shuffle the data
	public void shuffleData(double[][] inputs, double[][] outputs) {
	    java.util.Random random = new java.util.Random();
	    for (int i = inputs.length - 1; i > 0; i--) {
	        int j = random.nextInt(i + 1);

	        double[] tempInput = inputs[i];
	        inputs[i] = inputs[j];
	        inputs[j] = tempInput;

	        double[] tempOutput = outputs[i];
	        outputs[i] = outputs[j];
	        outputs[j] = tempOutput;
	    }
	}

}


