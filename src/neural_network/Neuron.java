package neural_network;

import java.util.List;


class Neuron {

    private List<Double> weightList;
    private List<Double> weighDelta;
    private double output;


    // Start work neuron
    void activation(List<Double> inputList) {

        Double input = sum(inputList);
        Double value = sigmoid(input);

        setOutput(value);
    }



    private Double sigmoid(double x) {
        return 1/(1 + Math.exp(-x));
    }

    // Summing all input connections
    private Double sum(List<Double> inputList) {

        double sum = 0.0;

        for (Double value: inputList)
            sum += value;

        return sum;
    }



    List<Double> getWeightList() {
        return weightList;
    }

    List<Double> getWeighDelta() {
        return weighDelta;
    }

    double getOutput() {
        return output;
    }

    void setWeightList(List<Double> weightList) {
        this.weightList = weightList;
    }

    void setWeighDelta(List<Double> weighDelta) {
        this.weighDelta = weighDelta;
    }

    void setOutput(double output) {
        this.output = output;
    }

}
