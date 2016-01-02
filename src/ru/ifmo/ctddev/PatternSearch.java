package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.function.Function;

public class PatternSearch extends OptimizationMethod {

    private final double[] init;
    private final double initStep, minStep;
    private final int iterations;

    /**
     *
     * @param init initial point of search
     * @param initStep initial step of exploration
     * @param minStep stopping criterion: minimal step of exploration
     * @param iterations iterations of exploratory search before pattern search
     */
    public PatternSearch(double[] init, double initStep, double minStep, int iterations) {
        this.init = Arrays.copyOf(init, init.length);
        this.initStep = initStep;
        this.minStep = minStep;
        this.iterations = iterations;
    }

    public PatternSearch() {
        this(new double[]{2, 2}, 1, 1e-4, 5);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        Point current = new Point(init);
        Point exploratoryNext = new Point(new double[arity]);
        Point patternNext = new Point(new double[arity]);
        current.quality = evaluator.apply(current.x);
        double step = initStep;
        while (step >= minStep) {
            // Exploratory search
            boolean improved = false;
            for (int i = 0; i < arity; i++) {
                exploratoryNext.x[i] = current.x[i];
            }
            exploratoryNext.quality = current.quality;
            for (int it = 0; it < iterations && step >= minStep; it++) {
                for (int i = 0; i < arity; i++) {
                    for (int k = 0; k < 2; k++) {
                        exploratoryNext.x[i] += (2 * k - 1) * step;
                        double newQuality = evaluator.apply(exploratoryNext.x);
                        if (newQuality < exploratoryNext.quality) {
                            exploratoryNext.quality = newQuality;
                            improved = true;
                        } else {
                            exploratoryNext.x[i] -= (2 * k - 1) * step;
                        }
                    }
                }
                if (!improved) {
                    step /= 2;
                }
            }
            // Pattern search
            int patternSteps = 1;
            double bestQuality = exploratoryNext.quality;
            while (improved) {
                for (int i = 0; i < arity; i++) {
                    patternNext.x[i] = current.x[i] + (patternSteps + 1) * (exploratoryNext.x[i] - current.x[i]);
                }
                patternNext.quality = evaluator.apply(patternNext.x);
                improved = patternNext.quality < bestQuality;
                if (improved) {
                    patternSteps++;
                    bestQuality = patternNext.quality;
                }
            }
            for (int i = 0; i < arity; i++) {
                current.x[i] = current.x[i] + patternSteps * (exploratoryNext.x[i] - current.x[i]);
            }
            current.quality = bestQuality;
        }
        return current;
    }

    @Override
    public String getName() {
        return "Pattern Search";
    }
}
