package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 * The optimization method based on the behaviour of flying particles attracted to some points.
 * Current realization is written according to Wikipedia article
 *
 * https://en.wikipedia.org/wiki/Particle_swarm_optimization
 *
 * You can learn more about tweaking parameters in the article
 *
 * http://hvass-labs.org/people/magnus/publications/pedersen10good-pso.pdf
 */
public class ParticleSwarm extends OptimizationMethod {

    private final double omega, phiP, phiG, diff;
    private final double[][] boundaries;
    private final int swarmSize;

    /**
     * Represents a flying particle.
     */
    protected static class Particle extends Point {
        /**
         * The particle's speed.
         */
        private double[] v;

        /**
         * The best point found by the article
         */
        private Point bestKnown;

        public Particle(double[] x, double[] v) {
            super(x);
            this.v = v;
            this.bestKnown = new Point(x);
        }
    }

    /**
     * Constructs a new particle swarm.
     * @param boundaries boundaries of the search space
     * @param swarmSize number of particles
     * @param omega decay of particles' speed over time
     * @param phiP attraction to the best local point
     * @param phiG attraction to the best global point
     * @param diff the stopping-criterion - method stops after the difference between the best and
     *             the worst solution is less than diff
     */
    public ParticleSwarm(double[][] boundaries, int swarmSize, double omega,
            double phiP, double phiG, double diff) {
        this.omega = omega;
        this.phiP = phiP;
        this.phiG = phiG;
        this.boundaries = boundaries;
        this.swarmSize = swarmSize;
        this.diff = diff;
    }

    /**
     * Default particle swarm constructor.
     */
    public ParticleSwarm() {
        this(new double[][]{{-8, 8}, {-8, 8}}, 4, 0.6, 0.4, 1.4, 0.1);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        Random r = new Random(System.currentTimeMillis());
        Particle[] swarm = new Particle[swarmSize];
        Point best = null;
        for (int i = 0; i < swarmSize; i++) {
            double[] x = new double[arity];
            double[] v = new double[arity];
            for (int j = 0; j < arity; j++) {
                x[j] = r.nextDouble() * (boundaries[j][1] - boundaries[j][0]) + boundaries[j][0];
                v[j] = (r.nextDouble() * 2 - 1) * (boundaries[j][1] - boundaries[j][0]);
            }
            swarm[i] = new Particle(x, v);
            swarm[i].quality = evaluator.apply(x);
            swarm[i].bestKnown.quality = swarm[i].quality;
            if (best == null || swarm[i].quality < best.quality) {
                best = swarm[i];
            }
        }
        double bestQuality = best.quality;
        best = new Point(best.x);
        best.quality = bestQuality;
        while (true) {
            double min = swarm[0].quality, max = swarm[0].quality;
            for (int i = 1; i < swarmSize; i++) {
                min = Math.min(min, swarm[i].quality);
                max = Math.max(max, swarm[i].quality);
            }
            if (max - min < diff) break;
            for (int i = 0; i < swarmSize; i++) {
                for (int j = 0; j < arity; j++) {
                    double rg = r.nextDouble(), rp = r.nextDouble();
                    swarm[i].v[j] = omega * swarm[i].v[j] + phiG * rg * (best.x[j] - swarm[i].x[j])
                            + phiP * rp * (swarm[i].bestKnown.x[j] - swarm[i].x[j]);
                }
                for (int j = 0; j < arity; j++) {
                    swarm[i].x[j] += swarm[i].v[j];
                }
                swarm[i].quality = evaluator.apply(swarm[i].x);
                if (swarm[i].quality < swarm[i].bestKnown.quality) {
                    swarm[i].bestKnown = new Point(swarm[i].x);
                    swarm[i].bestKnown.quality = swarm[i].quality;
                }
                if (swarm[i].quality < best.quality) {
                    best = new Point(swarm[i].x);
                    best.quality = swarm[i].quality;
                }
            }
        }
        Arrays.sort(swarm);
        return swarm[0];
    }

    @Override
    public String getName() {
        return "Particle swarm";
    }
}
