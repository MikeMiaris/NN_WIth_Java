
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class Neuron {
	
	private List<Double> inputs = new ArrayList<>();  //eiserxomena inputs
	private List<Double> weights = new ArrayList<>(); //IMPORTAMT: H lista ayth antiprosopeyei ta weights ta opoia EISERXONTAI ston neyrona
	private double output;
	private double delta; //error
	private double bias;
	private int function; //activation functions: 0:no activation 1:sigmoid, 2:Tanh, 3:RELU
	private int length;  //used for initializing nd parsing through inputs and weights list
	private double derivative; //paragogos
	
	//Constructor
	//Ylopoiei kai thn arxikopoihsh ton neuronon toy input layer me tis katallhes synthikes
	public Neuron(int function, int inputsNum){
		Random rand = new Random();
		this.function = function;
		//System.out.println("Activation function "+this.function);
		
		if(function == 0 ) { 
			this.bias = 0.0;
			
		}else {
			this.bias = (rand.nextDouble()*2)-1;
		}
		this.length = inputsNum;
		
		if (function != 0) {
			for(int i=0; i<inputsNum; i++) {
	            double x = rand.nextDouble() * 2 - 1;
				weights.add(x);
			}
		}

	}

	//Getters and Setters
	public double getDelta() {
		return delta;
	}
	
	public void setDelta(double delta) {
		this.delta = delta;
	}
	
    public List<Double> getInputs(){
        return inputs;
    }

    public void addInput(double input){
        inputs.add(input);
        this.length = inputs.size(); // Update length when inputs are set
    }

    public List<Double> getWeights(){
        return weights;
    }
    
    public double getWeights(int i) {
    	return weights.get(i);
    }

    public void setWeights(List<Double> weights){
        this.weights = weights;
    }

    public double getOutput(){
        return output;
    }

    public void setOutput(double output){
        this.output = output;
    }

    public double getBias(){
        return bias;
    }

    public void setBias(double bias){
        this.bias = bias;
    }

    public int getFunction(){
        return function;
    }

    public void setFunction(int function){
        this.function = function;
    }

    public int getLength(){
        return length;
    }

    public void setLength(int length){
        this.length = length;
    }
    
    public double getDerivative() {
    	return this.derivative;
    }
    
	
	//Calculate output of Neuron
	public void calculate() {
		double sum = 0;
		double output = 0;
		
		if(this.function == 0) {
			output = inputs.get(0);
		}
		else {
			for(int i = 0; i<this.length; i++) {
				
				sum += inputs.get(i)*weights.get(i);
			}
		
			sum += bias;

			if (this.function == 1) { //SIGMOID FUNCTION
				output = sigmoid(sum) ;
				//System.out.println("SIGMOID");
			}
			else if(this.function == 2) { //TANH FUNCTION
				output = tanh(sum);
				//System.out.println("TANH");
			}
			else if(this.function == 3) { //RELU FUNCTION
				output = relu(sum);
				//System.out.println("RELU");
			}
		}
					
		//upologismos derivative gia xrhsh sto backpropagation
		if (this.function == 1) { //SIGMOID derivative
			this.derivative = sigmoidDerivative(sum);
		}
		else if(this.function == 2) { //TANH derivative
			this.derivative = tanhDerivative(sum);
		}
		else if(this.function == 3) { //RELU derivative
			this.derivative = reluDerivative(sum);
		}
		
		
		this.output = output;
	}
	
	//Activation functions
    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
    
    public double tanh(double x){
        return Math.tanh(x);
    }
    
    public double relu(double x){
        return Math.max(0, x);
    }
	
    //Paragogoi ton parapano synartiseon energopoihshs
    public double sigmoidDerivative(double x) {
        return output * (1 - x);
    }
    
    public double tanhDerivative(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }
    
    public double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }
    
}

