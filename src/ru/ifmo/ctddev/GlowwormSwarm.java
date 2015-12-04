package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * The optimization method based on the behaviour of glowworms.
 * Current realization is written according to the article
 *
 * http://www.naturalspublishing.com/files/published/8xv0gm5044k95s.pdf
 *
 * Levy flights are changed to uniform random steps for simplicity.
 */
public class GlowwormSwarm extends OptimizationMethod {

    /**
     * Represents a glowworm.
     */
    protected class Glowworm extends Point {
        /**
         * Luciferin is used for attracting neighbour glowworms.
         */
        public double luciferin = initialLuciferin;

        /**
         * Visibility is used for determining whether other glowworm is our neighbour.
         */
        protected double visibility;

        public Glowworm(double[] x) {
            super(x);
            this.visibility = inititalVisibility;
        }

        /**
         * Returns euclidean distance to other glowworm.
         * @param other other glowworm
         * @return euclidean distance
         */
        public double distTo(Glowworm other) {
            double sqrDist = 0;
            for (int i = 0; i < x.length; i++) {
                sqrDist += (x[i] - other.x[i]) * (x[i] - other.x[i]);
            }
            return Math.sqrt(sqrDist);
        }

        /**
         * Updates the glowworm's luciferin.
         * @param gamma coefficient of converting current quality to luciferin
         * @param luciferinDecay decay of previously accumulated luciferin
         */
        public void updateLuciferin(double gamma, double luciferinDecay) {
            luciferin = luciferin * (1 - luciferinDecay) + gamma * quality;
        }

        /**
         * Returns the list of better visible neighbours.
         * @param swarm all glowworms
         * @return neighbours
         */
        protected List<Glowworm> getNeighbours(Glowworm[] swarm) {
            List<Glowworm> neighbours = new ArrayList<>();
            for (Glowworm g: swarm) if (g.quality < quality) {
                double d = distTo(g);
                if (d > 0 && d <= visibility) {
                    neighbours.add(g);
                }
            }
            return neighbours;
        }

        /**
         * Selects one of the brighter (having more luciferin) neighbours
         * with the probability proportional to the luciferin difference.
         * @param neighbours neighbour glowworms
         * @return the selected neighbour
         */
        protected Glowworm selectNeighbour(List<Glowworm> neighbours) {
            double sum = 0;
            for (Glowworm g: neighbours) {
                sum += quality - g.quality;
            }
            double q = r.nextDouble() * sum;
            int j = 0;
            sum = 0;
            while (j < neighbours.size() - 1 && sum + quality - neighbours.get(j).quality < q) {
                sum += quality - neighbours.get(j).quality;
                j++;
            }
            if (neighbours.size() > 0) {
                return neighbours.get(j);
            } else {
                return null;
            }
        }

        /**
         * Moves to the better neighbour glowworm or does a random step if there are no such glowworms.
         * @param neighbour better neighbour glowworm or null if there aren't such.
         * @param step size of the step to the neighbour
         */
        public void moveToNeighbour(Glowworm neighbour, double step) {
            if (neighbour != null) {
                double d = distTo(neighbour);
                for (int i = 0; i < x.length; i++) {
                    x[i] += step * (neighbour.x[i] - x[i]) / d;
                }
            } else {
                double[] m = randomMove(x.length, step);
                for (int i = 0; i < x.length; i++) {
                    x[i] += m[i];
                }
            }
        }

        /**
         * Uniform random step made by the glowworm.
         * @param arity dimensionality of the search space
         * @param step maximum size of the step in each dimension
         * @return random step
         */
        protected double[] randomMove(int arity, double step) {
            double[] result = new double[arity];
            for (int i = 0; i < arity; i++) {
                result[i] = (r.nextDouble() * 2 - 1) * step / 2;
            }
            return result;
        }

        /**
         * Updates visibility of the glowworm.
         * @param neighbour list of neighbours
         * @param beta estimate of the proportionality between visibility radius and number of visible neighbours
         * @param nt estimate of required number of visible neighbours
         */
        public void updateVisibility(List<Glowworm> neighbour, double beta, double nt) {
            visibility = Math.min(maxVisibility, Math.max(visibility, beta * (nt - neighbour.size())));
        }
    }

    private final double initialLuciferin, inititalVisibility, maxVisibility, step, nt, luciferinDecay, beta, gamma, diff;
    private final double[][] boundaries;
    private final int swarmSize, maxIterations;
    private Random r;

    /**
     * Constructs a new glowworm swarm.
     * @param boundaries boundaries of the search space
     * @param swarmSize number of glowworms
     * @param maxIterations maximum number of iterations over the swarm
     * @param initialLuciferin initial luciferin of glowworms
     * @param inititalVisibility initial visibility of glowworms
     * @param maxVisibility maximum visibility of glowworms
     * @param step size of step towards neighbour glowworm
     * @param nt estimate of required number of visible neighbours
     * @param beta estimate of the proportionality between visibility radius and number of visible neighbours
     * @param luciferinDecay decay of previously accumulated luciferin
     * @param gamma coefficient of converting current quality to luciferin
     * @param diff stopping-criterion - the method stops after the difference between the best and
     *             the worst solution is less than diff
     */
    public GlowwormSwarm(double[][] boundaries, int swarmSize, int maxIterations,
            double initialLuciferin, double inititalVisibility, double maxVisibility,
            double step, double nt, double beta,
            double luciferinDecay,double gamma, double diff) {
        this.initialLuciferin = initialLuciferin;
        this.inititalVisibility = inititalVisibility;
        this.maxVisibility = maxVisibility;
        this.step = step;
        this.nt = nt;
        this.luciferinDecay = luciferinDecay;
        this.beta = beta;
        this.gamma = gamma;
        this.boundaries = boundaries;
        this.swarmSize = swarmSize;
        this.maxIterations = maxIterations;
        this.diff = diff;
    }

    /**
     * Default glowworm swarm constructor.
     */
    public GlowwormSwarm() {
        this(new double[][]{{-8, 8}, {-8, 8}}, 20, 15, 0, 2, 8, 0.1, 5, 2, 0.5, 1, 0.1);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        r = new Random(System.currentTimeMillis());
        Glowworm[] swarm = new Glowworm[swarmSize];
        for (int i = 0; i < swarmSize; i++) {
            double[] x = new double[arity];
            for (int j = 0; j < arity; j++) {
                x[j] = r.nextDouble() * (boundaries[j][1] - boundaries[j][0]) + boundaries[j][0];
            }
            swarm[i] = new Glowworm(x);
            swarm[i].quality = evaluator.apply(x);
        }
        for (int it = 0; it < maxIterations; it++) {
            double min = swarm[0].quality, max = swarm[0].quality;
            for (int i = 1; i < swarmSize; i++) {
                min = Math.min(min, swarm[i].quality);
                max = Math.max(max, swarm[i].quality);
            }
            if (max - min < diff) break;
            for (Glowworm g : swarm) {
                List<Glowworm> neighbours = g.getNeighbours(swarm);
                g.updateVisibility(neighbours, beta, nt);
                g.moveToNeighbour(g.selectNeighbour(neighbours), step);
            }
            for (int i = 0; i < swarmSize; i++) {
                swarm[i].quality = evaluator.apply(swarm[i].x);
                swarm[i].updateLuciferin(gamma, luciferinDecay);
            }
        }
        Arrays.sort(swarm);
        return swarm[0];
    }

    @Override
    public String getName() {
        return "Glowworm swarm";
    }
}
