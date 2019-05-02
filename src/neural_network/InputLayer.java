package neural_network;

import java.util.ArrayList;
import java.util.List;


class InputLayer extends Layer {

    InputLayer(int n) {
        super(n);
    }


    // Activation all layer neurons
    List<Double> start(List<Double> layerInput) {

        List<Double> nextLayerInput = new ArrayList<>();

        for (int i = 0; i < layerSize(); i++) {

            Neuron neuron = getNeuronList().get(i);
            neuron.setOutput(layerInput.get(i));

            // form next layer input values
            List<Double> list = jump(neuron);
            nextLayerInput.addAll(list);
        }


        return nextLayerInput;
    }


    /**
     * Change neurons weight (learning)
     * @param e learning speed
     * @param a moment
     * @param deltaList list of up-layer neurons delta
     */
    void backPropagation(double e, double a, List<Double> deltaList) {

        for (Neuron neuron : getNeuronList())
            weightUpdate(neuron, e, a, deltaList);
    }


}
