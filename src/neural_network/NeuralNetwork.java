package neural_network;

import java.util.ArrayList;
import java.util.List;


public class NeuralNetwork {

    private double[][] weight = {{0.45, 0.78, -0.12, 0.13}, {1.5, -2.3}, {1}}; // connection between neurons

    private Layer inputLayer;
    private Layer hiddenLayer;
    private Layer outputLayer;

    // Create Neuron Network
    // XOR: 3 Layers (input: 2n, hidden: 2n, output: 1n)
    public NeuralNetwork() {

        inputLayer = new Layer(2);
        hiddenLayer = new Layer(2);
        outputLayer = new Layer(1);

        inputLayer.init(weight[0]);
        hiddenLayer.init(weight[1]);
        outputLayer.init(weight[2]);
    }


    public void processing(double a, double b, double c) {

        List<Double> outValues; // layer output values
        List<Double> inValues = new ArrayList<>();  // initial input values
        inValues.add(a);
        inValues.add(b);

        outValues = inputLayer.start(inValues, 1, true);
        outValues = hiddenLayer.start(outValues, inputLayer.layerSize());
        double result = outputLayer.start(outValues, hiddenLayer.layerSize()).get(0);

        System.out.println(result);
    }



    public Layer getInputLayer() {
        return inputLayer;
    }

    public Layer getHiddenLayer() {
        return hiddenLayer;
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }

    public double[][] getWeight() {
        return weight;
    }

}
