package neural_network;


class OutputLayer extends Layer {

    OutputLayer(int n) {
        super(n);
    }


    double delta(double result, double ideal) {
        return (ideal - result) * ((1 - result) * result);
    }

}
