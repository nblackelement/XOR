package neural_network;

import java.util.ArrayList;
import java.util.List;


class Layer {

    private List<Neuron> neuronList = new ArrayList<>();

    Layer (int n) {

        // Neurons creating
        for (int i = 0; i < n; i++) {

            Neuron neuron = new Neuron();
            neuronList.add(neuron);
        }
    }


    /**
     * Connect neurons between this layer and next
     * @param weight all weights of layer
     */
    void init(double[] weight) {

        int connectNum = weight.length / layerSize();    // amount of out connections in neuron

        for (int i = 0; i < layerSize(); i++) {

            List<Double> weightList = new ArrayList<>();
            List<Double> deltaList = new ArrayList<>();

            // Formation weights & delta for neuron
            for (int j = 0; j < connectNum; j++) {

                weightList.add(weight[i * connectNum + j]);
                deltaList.add(0.0);
            }

            getNeuronList().get(i).setWeightList(weightList);
            getNeuronList().get(i).setWeighDelta(deltaList);
        }

    }


    // Activation all layer neurons
    List<Double> start(List<Double> layerInput, int previousLayerSize) {

        int inputNum = layerInput.size() / previousLayerSize;   // amount of input connections in neuron
        List<Double> nextLayerInput = new ArrayList<>();

        for (int i = 0; i < layerSize(); i++) {

            Neuron neuron = getNeuronList().get(i);
            List<Double> inputList = new ArrayList<>();

            for (int j = 0; j < layerInput.size(); j += inputNum)
                inputList.add(layerInput.get(i + j));

            neuron.activation(inputList);

            // form next layer input values
            List<Double> list = jump(neuron);
            nextLayerInput.addAll(list);
        }


        return nextLayerInput;
    }



    // jump through connection
    List<Double> jump(Neuron neuron) {

        List<Double> weightList = neuron.getWeightList();
        double output = neuron.getOutput();

        // jump all neuron out-connection
        List<Double> nextInput = new ArrayList<>();

        for (Double weight : weightList)
            nextInput.add(weight * output);


        return nextInput;
    }


    // Function of Back Propagation Method
    void weightUpdate(Neuron neuron, double e, double a, List<Double> deltaList) {

        for (int i = 0; i < neuron.getWeightList().size(); i++) {

            double grad = gradient(neuron.getOutput(), deltaList.get(i));
            double newWeightDelta = e * grad + a * neuron.getWeighDelta().get(i);
            double newWeight = neuron.getWeightList().get(i) + newWeightDelta;

            neuron.getWeightList().set(i, newWeight);
            neuron.getWeighDelta().set(i, newWeightDelta);
        }

    }


    private double gradient(double output, double delta) {
        return output * delta;
    }



    List<Neuron> getNeuronList() {
        return neuronList;
    }

    int layerSize() {
        return getNeuronList().size();
    }

}
