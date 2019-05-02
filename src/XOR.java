import neural_network.NeuralNetwork;


public class XOR {

    public static void main(String[] args) {

        NeuralNetwork neuralNetwork = new NeuralNetwork();
        double result;
        double error;

        // Test Era
        for (int i = 0; i < 2; i++) {

            result = neuralNetwork.processing(1, 0, 1);
            error = neuralNetwork.error(i + 1);
            answerOutput(result, error);
        }
        neuralNetwork.eraEnd();


        // learning network
        learning(neuralNetwork, 10000);

        // Check
        result = neuralNetwork.processing(1, 0, 1);
        error = neuralNetwork.error(1);
        answerOutput(result, error);

    }



    private static void answerOutput(double result, double error) {

        System.out.println(result + ", error: " + error + "%");
    }


    private static void learning(NeuralNetwork neuralNetwork, int iterationNum) {

        // Learning Era
        System.out.print("\nlearning...");

        for (int i = 0; i < iterationNum; i++) {

            neuralNetwork.processing(0, 0, 0);
            neuralNetwork.processing(0, 1, 1);
            neuralNetwork.processing(1, 0, 1);
            neuralNetwork.processing(1, 1, 0);
        }

        double error = neuralNetwork.error(iterationNum);
        System.out.println(" , error ~ " + error + "\n");

        neuralNetwork.eraEnd();
    }


}
