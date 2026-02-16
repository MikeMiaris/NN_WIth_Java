import java.util.ArrayList;
import java.util.List;

public class Layer {
	
	List<Neuron> neurons = new ArrayList<>();

	//constructor
	public Layer(int num, int function, int inputsNum) {
		for (int i = 0; i<num; i++) {
			Neuron neuron = new Neuron(function,inputsNum);
			neurons.add(neuron);
		}	
	}
	
	//neurons getter
	public List<Neuron> getNeurons(){
		return this.neurons;
	}
	
	//neurons list size getter
	public int getSize() {
		return this.neurons.size();
	}
		
}
