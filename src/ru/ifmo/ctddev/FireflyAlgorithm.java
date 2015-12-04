package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 * The optimization method based on the behaviour of fireflies.
 * Current realization is written according to the article
 *
 * http://arxiv.org/pdf/1003.1464.pdf
 */
public class FireflyAlgorithm extends OptimizationMethod {

    /**
     * Represents a firefly.
     */
    protected static class Firefly extends Point {
        public Firefly(double[] x) {
            super(x);
        }

        /**
         * Returns euclidean distance to other firefly.
         * @param other other firefly
         * @return euclidean distance
         */
        public double distTo(Firefly other) {
            double sqrDist = 0;
            for (int i = 0; i < x.length; i++) {
                sqrDist += (x[i] - other.x[i]) * (x[i] - other.x[i]);
            }
            return Math.sqrt(sqrDist);
        }
    }

    private final double alpha, beta, gamma, diff;
    private final double[][] boundaries;
    private final int swarmSize, maxIterations;
    private Random r;

    /**
     * Uniform random step made by firefly.
     * @param arity dimensionality of the search space
     * @param alpha maximum size of the step in each dimension
     * @return random step
     */
    protected double[] randomMove(int arity, double alpha) {
        double[] result = new double[arity];
        for (int i = 0; i < arity; i++) {
            result[i] = (r.nextDouble() * 2 - 1) * alpha;
        }
        return result;
    }

    /**
     * Construct a new firefly algorithm.
     * @param boundaries boundaries of the search space
     * @param swarmSize number of fireflies
     * @param maxIterations maximum number of iterations over the swarm
     * @param alpha maximum size of a random step
     * @param beta coefficient of attraction
     * @param gamma exponential decay of attraction over distance
     * @param diff stopping-criterion - the method stops after the difference between the best and
     *             the worst solution is less than diff
     */
    public FireflyAlgorithm(double[][] boundaries, int swarmSize, int maxIterations,
            double alpha, double beta, double gamma, double diff) {
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.boundaries = boundaries;
        this.swarmSize = swarmSize;
        this.maxIterations = maxIterations;
        this.diff = diff;
    }

    /**
     * Default firefly algorithm constructor.
     */
    public FireflyAlgorithm() {
        this(new double[][] {{-8, 8}, {-8, 8}}, 20, 15, 0.1, 1, 1, 0.1);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        r = new Random(System.currentTimeMillis());
        Firefly[] swarm = new Firefly[swarmSize];
        for (int i = 0; i < swarmSize; i++) {
            double[] x = new double[arity];
            for (int j = 0; j < arity; j++) {
                x[j] = r.nextDouble() * (boundaries[j][1] - boundaries[j][0]) + boundaries[j][0];
            }
            swarm[i] = new Firefly(x);
            swarm[i].quality = evaluator.apply(x);
        }
        for (int it = 0; it < maxIterations; it++) {
            double min = swarm[0].quality, max = swarm[0].quality;
            for (int i = 1; i < swarmSize; i++) {
                min = Math.min(min, swarm[i].quality);
                max = Math.max(max, swarm[i].quality);
            }
            if (max - min < diff) break;
            for (int i = 0; i < swarmSize; i++) {
                for (int j = 0; j <= i; j++) {
                    Firefly brighter = swarm[i], darker = swarm[j];
                    if (swarm[i].quality > swarm[j].quality) {
                        brighter = swarm[j];
                        darker = swarm[i];
                    }
                    double attraction = Math.exp(-gamma * swarm[i].distTo(swarm[j])) * beta;
                    double[] move = randomMove(arity, alpha);
                    for (int k = 0; k < arity; k++) {
                        darker.x[k] += attraction * (brighter.x[k] - darker.x[k]) + move[k];
                        darker.x[k] = Math.min(boundaries[k][1], Math.max(darker.x[k], boundaries[k][0]));
                    }
                }
            }
            for (int i = 0; i < swarmSize; i++) {
                swarm[i].quality = evaluator.apply(swarm[i].x);
            }
        }
        Arrays.sort(swarm);
        return swarm[0];
    }

    @Override
    public String getName() {
        return "Firefly algorithm";
    }
}
