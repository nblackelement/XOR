package neural_network;

import javax.management.BadAttributeValueExpException;
import java.util.ArrayList;
import java.util.List;


class Neuron {

    private List<Double> weightList;
    private List<Double> outputList;


    /**
     * Start work neuron
     * @param inputList list of neuron input values
     */
    void activation(List<Double> inputList) {

        // calculation
        Double inValue = sum(inputList);
        Double outValue = sigmoid(inValue);
        outputList = new ArrayList<>();

        for (Double weight: weightList)
            outputList.add(outValue * weight);
    }

    /**
     * Overload function for inputLayer
     * @throws BadAttributeValueExpException if isInputLayer: false
     */
    void activation(List<Double> inputList, boolean isInputLayer) throws BadAttributeValueExpException {

        if (isInputLayer) {

            double outValue = inputList.get(0);
            outputList = new ArrayList<>();

            for (Double weight: weightList)
                outputList.add(outValue * weight);

        } else {
            throw new BadAttributeValueExpException(isInputLayer);
        }
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



    void setWeightList(List<Double> weightList) {
        this.weightList = weightList;
    }

    List<Double> getOutputList() {
        return outputList;
    }

}
