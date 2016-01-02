package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class LocalUnimodalSampling extends OptimizationMethod {
    private final double[] init;
    private final double alpha, initRange, minRange;
    private final Random r = new Random();

    public LocalUnimodalSampling(double[] init, double alpha, double initRange, double minRange) {
        this.init = Arrays.copyOf(init, init.length);
        this.alpha = alpha;
        this.initRange = initRange;
        this.minRange = minRange;
    }

    public LocalUnimodalSampling() {
        this(new double[]{2, 2}, 0.3, 1, 1e-4);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        Point current = new Point(init);
        Point newPoint = new Point(init);
        current.quality = evaluator.apply(current.x);
        double range = initRange;
        while (range > minRange) {
            for (int i = 0; i < arity; i++) {
                newPoint.x[i] = current.x[i] + (2 * r.nextDouble() - 1) * range;
            }
            newPoint.quality = evaluator.apply(newPoint.x);
            if (newPoint.quality < current.quality) {
                for (int i = 0; i < arity; i++) {
                    current.x[i] = newPoint.x[i];
                }
                current.quality = newPoint.quality;
            } else {
                range *= Math.pow(2, -alpha / arity);
            }
        }
        return current;
    }

    @Override
    public String getName() {
        return "Local Unimodal Sampling";
    }
}
