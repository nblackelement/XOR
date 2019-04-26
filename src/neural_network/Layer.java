package neural_network;

import javax.management.BadAttributeValueExpException;
import java.util.ArrayList;
import java.util.List;


class Layer {

    private List<Neuron> neuronList = new ArrayList<>();


    Layer (int nAmount) {

        // Neurons creating
        for (int i = 0; i < nAmount; i++) {

            Neuron neuron = new Neuron();
            neuronList.add(neuron);
        }
    }


    /**
     * Connect neurons between this layer and next
     * @param weight all weights of layer
     */
    void init(double[] weight) {

        int wAmount = weight.length / layerSize();    // amount of out connections in neuron

        for (int i = 0; i < layerSize(); i++) {

            List<Double> weightList = new ArrayList<>();

            // Formation weights for neuron
            for (int j = 0; j < wAmount; j++)
                weightList.add(weight[i * wAmount + j]);

            neuronList.get(i).setWeightList(weightList);
        }
    }


    /**
     * Activation all layer neurons
     *
     * @param inValues list of input values
     * @param previousLayerSize for calc inAmount
     * @param isInputLayer for correct handle initial values
     * @return list of all neurons output values
     */
    List<Double> start(List<Double> inValues, int previousLayerSize, boolean isInputLayer) {

        int inAmount = inValues.size() / previousLayerSize;   // amount of input connections in neuron
        List<Double> outValues = new ArrayList<>();

        // layer calculation
        for (int i = 0; i < layerSize(); i++) {

            Neuron neuron = neuronList.get(i);
            List<Double> inputList = new ArrayList<>();

            for (int j = 0; j < inValues.size(); j += inAmount)
              inputList.add(inValues.get(i + j));

            if (isInputLayer) {
                try {
                    neuron.activation(inputList, true);
                } catch (BadAttributeValueExpException e) {
                    e.printStackTrace();
                }
            } else {
                neuron.activation(inputList);
            }

            List<Double> tmp = neuron.getOutputList();
            outValues.addAll(tmp);
        }

        return outValues;
    }

    // Overload function for convenience
    List<Double> start(List<Double> inValue, int previousLayerSize) {

        boolean isInputLayer = false;
        return start(inValue, previousLayerSize, isInputLayer);
    }



    int layerSize() {
        return neuronList.size();
    }

    List<Neuron> getNeuronList() {
        return neuronList;
    }

}
