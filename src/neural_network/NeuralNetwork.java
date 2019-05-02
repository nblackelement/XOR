package neural_network;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;


public class NeuralNetwork {

    private double[][] weight = {{0.45, 0.78, -0.12, 0.13}, {1.5, -2.3}, {1}}; // connection between neurons
    private double e = 0.7; // learning speed default value
    private double a = 0.3; // moment default value

    private double mse = 0; // error sum of era

    private InputLayer inputLayer;
    private HiddenLayer hiddenLayer;
    private OutputLayer outputLayer;

    // Create Neuron Network
    // XOR: 3 Layers (input: 2n, hidden: 2n, output: 1n)
    public NeuralNetwork() {

        inputLayer = new InputLayer(2);
        hiddenLayer = new HiddenLayer(2);
        outputLayer = new OutputLayer(1);
        inputLayer.init(weight[0]);
        hiddenLayer.init(weight[1]);
        outputLayer.init(weight[2]);
    }


    public double processing(double a, double b, double c) {

        double result;
        List<Double> inputList = new ArrayList<>();    // layer input values
        inputList.add(a);
        inputList.add(b);

        // Layers Calculation
        inputList = inputLayer.start(inputList);
        inputList = hiddenLayer.start(inputList, getInputLayer().layerSize());
        inputList = outputLayer.start(inputList, getHiddenLayer().layerSize());

        result = inputList.get(0);
        correction(result, c);

        // MSE - Mean Squared Error
        mse += Math.pow(result - c, 2);

        return result;
    }



    /**
     * Calculating error on era
     * @param n num of era iteration
     */
    public double error(int n) {

        double error = (getMse() / n) * 100;    // in percents
        return new BigDecimal(error).setScale(1, RoundingMode.HALF_DOWN).doubleValue();
    }


    // Correcting weights (learning)
    private void correction(double result, double ideal) {

        // Output Layer
        double tmp = outputLayer.delta(result, ideal);
        List<Double> deltaList = new ArrayList<>();
        deltaList.add(tmp);

        // Hidden Layers
        deltaList = hiddenLayer.backPropagation(e, a, deltaList);

        // Input Layer
        inputLayer.backPropagation(e, a, deltaList);

    }



    private InputLayer getInputLayer() {
        return inputLayer;
    }

    private HiddenLayer getHiddenLayer() {
        return hiddenLayer;
    }

    private OutputLayer getOutputLayer() {
        return outputLayer;
    }

    public double[][] getWeight() {
        return weight;
    }

    public double getE() {
        return e;
    }

    public double getA() {
        return a;
    }

    private double getMse() {
        return mse;
    }

    public void setE(double e) {
        this.e = e;
    }

    public void setA(double a) {
        this.a = a;
    }

    public void eraEnd() {
        this.mse = 0;
    }

}
