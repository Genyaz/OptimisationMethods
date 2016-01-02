package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class SimulatedAnnealing extends OptimizationMethod {

    private final double randomStep, initTemperature, cooling;
    private final double[] init;
    private final Random r = new Random();

    public SimulatedAnnealing(double[] init, double randomStep, double initTemperature, double cooling) {
        this.init = Arrays.copyOf(init, init.length);
        this.randomStep = randomStep;
        this.initTemperature = initTemperature;
        this.cooling = cooling;
    }

    public SimulatedAnnealing() {
        this(new double[]{2, 2}, 0.1, 1, 0.005);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        Point current = new Point(init);
        Point newPoint = new Point(init);
        current.quality = evaluator.apply(current.x);
        double temperature = initTemperature;
        while (temperature > 0) {
            for (int i = 0; i < arity; i++) {
                newPoint.x[i] = current.x[i] + (2 * r.nextDouble() - 1) * randomStep;
            }
            newPoint.quality = evaluator.apply(newPoint.x);
            if (r.nextDouble() < Math.exp((current.quality - newPoint.quality) / temperature)) {
                for (int i = 0; i < arity; i++) {
                    current.x[i] = newPoint.x[i];
                }
                current.quality = newPoint.quality;
            }
            temperature -= cooling;
        }
        return current;
    }

    @Override
    public String getName() {
        return "Simulated Annealing";
    }
}
