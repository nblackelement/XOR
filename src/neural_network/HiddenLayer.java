package neural_network;

import java.util.ArrayList;
import java.util.List;


class HiddenLayer extends Layer {

    HiddenLayer(int n) {
        super(n);
    }


    private double delta(Neuron neuron, List<Double> delta) {

        List<Double> weightList = neuron.getWeightList();
        double output = neuron.getOutput();
        double sum = 0;

        for (int i = 0; i < weightList.size(); i++)
            sum += weightList.get(i) * delta.get(i);


        return ((1 - output) * output) * sum;
    }


    /**
     * Change neurons weight (learning)
     * @param e learning speed
     * @param a moment
     * @param deltaList list of up-layer neurons delta
     * @return list of neurons delta values for down-layer
     */
    List<Double> backPropagation(double e, double a, List<Double> deltaList) {

        List<Double> currentDeltaList = new ArrayList<>();  // for previous layer

        for (Neuron neuron : getNeuronList()) {

            weightUpdate(neuron, e, a, deltaList);

            double delta = delta(neuron, deltaList);
            currentDeltaList.add(delta);
        }


        return currentDeltaList;
    }


}
