import neural_network.NeuralNetwork;


public class XOR {
    public static void main(String[] args) {

        NeuralNetwork neuralNetwork = new NeuralNetwork();

        neuralNetwork.processing(1, 0, 1);
        neuralNetwork.error(1);

//        neuralNetwork.processing(0, 0, 0);
//        neuralNetwork.processing(0, 1, 1);
//        neuralNetwork.processing(1, 1, 0);

    }


}
