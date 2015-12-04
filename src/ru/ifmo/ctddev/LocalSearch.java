package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.function.Function;

/**
 * The optimization method based on moving along axes with the given step.
 */
public class LocalSearch extends OptimizationMethod {
    private double step;
    private double[] init;


    /**
     * Constructs a new local search method.
     * @param step size of step in any dimension
     * @param init initial point in the search space
     */
    public LocalSearch(double step, double[] init) {
        this.init = Arrays.copyOf(init, init.length);
        this.step = step;
    }

    /**
     * Default local search constructor.
     */
    public LocalSearch() {
        this(0.1, new double[]{2, 2});
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        Point best = new Point(init);
        best.quality = evaluator.apply(init);
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < arity; i++) {
                best.x[i] += step;
                Point plusStep = new Point(best.x);
                plusStep.quality = evaluator.apply(plusStep.x);
                best.x[i] -= 2 * step;
                Point minusStep = new Point(best.x);
                minusStep.quality = evaluator.apply(minusStep.x);
                best.x[i] += step;
                if (plusStep.quality < best.quality) {
                    best = plusStep;
                    improved = true;
                }
                if (minusStep.quality < best.quality) {
                    best = minusStep;
                    improved = true;
                }
            }
        }
        return best;
    }

    @Override
    public String getName() {
        return "Local search";
    }
}
