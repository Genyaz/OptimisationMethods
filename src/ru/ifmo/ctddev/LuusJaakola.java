package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class LuusJaakola extends OptimizationMethod {
    private final double[] init;
    private final double initRange, minRange;
    private final Random r = new Random();

    public LuusJaakola(double[] init, double initRange, double minRange) {
        this.init = Arrays.copyOf(init, init.length);
        this.initRange = initRange;
        this.minRange = minRange;
    }

    public LuusJaakola() {
        this(new double[]{2, 2}, 1, 1e-4);
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
                range *= 0.95;
            }
        }
        return current;
    }

    @Override
    public String getName() {
        return "Luus-Jaakola";
    }
}
