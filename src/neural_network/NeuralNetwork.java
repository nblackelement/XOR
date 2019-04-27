package neural_network;

import java.util.ArrayList;
import java.util.List;


public class NeuralNetwork {

    private double[][] weight = {{0.45, 0.78, -0.12, 0.13}, {1.5, -2.3}, {1}}; // connection between neurons

    private double mse = 0; // era sum

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


    public double processing(double a, double b, double c) {

        List<Double> outValues; // layer output values
        List<Double> inValues = new ArrayList<>();  // initial input values
        inValues.add(a);
        inValues.add(b);

        outValues = inputLayer.start(inValues, 1, true);
        outValues = hiddenLayer.start(outValues, inputLayer.layerSize());
        double result = outputLayer.start(outValues, hiddenLayer.layerSize()).get(0);

        // MSE - Mean Squared Error
        mse += Math.pow(result - c, 2);

        return result;
    }


    /**
     * Calculating error on era
     * @param n num of era iteration
     */
    public void error(int n) {

        double error = (mse / n) * 100;
        setMse(0);

        System.out.printf("%.1f%c", error, '%');
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

    public double getMse() {
        return mse;
    }

    public void setMse(double mse) {
        this.mse = mse;
    }

}
