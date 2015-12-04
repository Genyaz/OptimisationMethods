package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;

/**
 * The optimization method based on the shrinking simplex.
 * Current realization is written according to Wikipedia article
 *
 * https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
 */
public class NelderMead extends OptimizationMethod {
    private final double alpha, gamma, p, sigma, eps, diff;
    private final double[][] init;
    private final int threads;

    /**
     * Constructs a new Nelder-Mead method.
     * @param alpha coefficient of reflection
     * @param gamma coefficient of expansion
     * @param p coeffictient of contraction
     * @param sigma coefficient of reduction
     * @param init initial simplex with (arity + 1) points, init[i][j] is the j-th coordinate of the i-th point
     * @param eps stopping-criterion - the method stops after
     *            the maximum distance between simplex's points is less than eps
     * @param diff stopping-criterion - the method stops after
     *             the difference between the best and the worst solution is less than diff
     * @param threads number of additional threads, should be >= 1
     */
    public NelderMead(double alpha, double gamma, double p,
                      double sigma, double[][] init, double eps, double diff, int threads) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.p = p;
        this.sigma = sigma;
        this.eps = eps;
        this.diff = diff;
        this.threads = threads;
        this.init = init;
    }

    /**
     * Default Nelder-Mead constructor.
     */
    public NelderMead() {
        this(1, 2, -0.5, 0.5, new double[][]{{-8, 8}, {-8, -8}, {16, 0}}, 0.001, 0.05, 8);
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        ExecutorService executorService = Executors.newFixedThreadPool(threads);
        final CyclicBarrier cyclicBarrier = new CyclicBarrier(arity + 1, () -> {});
        Point[] points = new Point[arity + 1];
        for (int i = 0; i < arity + 1; i++) {
            double[] x = Arrays.copyOf(init[i], arity);
            points[i] = new Point(x);
        }
        for (int i = 0; i < arity; i++) {
            final Point p = points[i];
            executorService.execute(() -> {
                p.quality = evaluator.apply(p.x);
                try {
                    cyclicBarrier.await();
                } catch (InterruptedException | BrokenBarrierException e) {
                    e.printStackTrace();
                }
            });
        }
        points[arity].quality = evaluator.apply(points[arity].x);
        try {
            cyclicBarrier.await();
        } catch (InterruptedException | BrokenBarrierException e) {
            e.printStackTrace();
            return null;
        }
        while (true) {
            Arrays.sort(points);
            if (points[arity].quality - points[0].quality < diff) break;
            double maxDist = 0, dist;
            for (int i = 0; i < arity + 1; i++) {
                for (int j = i + 1; j < arity + 1; j++) {
                    dist = 0;
                    for (int k = 0; k < arity; k++) {
                        dist += (points[i].x[k] - points[j].x[k]) * (points[i].x[k] - points[j].x[k]);
                    }
                    if (dist > maxDist) {
                        maxDist = dist;
                    }
                }
            }
            if (maxDist < eps * eps) break;
            double[] cm = new double[arity];
            for (int i = 0; i < arity; i++) {
                for (int j = 0; j < arity; j++) {
                    cm[j] += points[i].x[j];
                }
            }
            for (int j = 0; j < arity; j++) {
                cm[j] /= arity;
            }
            double[] refl = new double[arity];
            for (int j = 0; j < arity; j++) {
                refl[j] = (1 + alpha) * points[arity].x[j] - alpha * cm[j];
            }
            Point reflected = new Point(refl);
            reflected.quality = evaluator.apply(refl);
            if (reflected.quality < points[arity - 1].quality && reflected.quality >= points[0].quality) {
                points[arity] = reflected;
                continue;
            }
            if (reflected.quality < points[0].quality) {
                double[] exp = new double[arity];
                for (int j = 0; j < arity; j++) {
                    exp[j] = (1 + gamma) * points[arity].x[j] - gamma * cm[j];
                }
                Point expanded = new Point(exp);
                expanded.quality = evaluator.apply(exp);
                if (expanded.quality < reflected.quality) {
                    points[arity] = expanded;
                } else {
                    points[arity] = reflected;
                }
                continue;
            }
            double[] cont = new double[arity];
            for (int j = 0; j < arity; j++) {
                cont[j] = (1 + p) * points[arity].x[j] - p * cm[j];
            }
            Point contracted = new Point(cont);
            contracted.quality = evaluator.apply(cont);
            if (contracted.quality < points[arity].quality) {
                points[arity] = contracted;
                continue;
            }
            double[] reductionCenter = points[0].x;
            for (int i = 1; i <= arity; i++) {
                for (int j = 0; j < arity; j++) {
                    points[i].x[j] = (1 - sigma) * reductionCenter[j] + sigma * points[i].x[j];
                }
            }
            for (int i = 1; i <= arity; i++) {
                final Point p = points[i];
                executorService.execute(() -> {
                    p.quality = evaluator.apply(p.x);
                    try {
                        cyclicBarrier.await();
                    } catch (InterruptedException | BrokenBarrierException e) {
                        e.printStackTrace();
                    }
                });
            }
            try {
                cyclicBarrier.await();
            } catch (InterruptedException | BrokenBarrierException e) {
                e.printStackTrace();
                return null;
            }
        }
        executorService.shutdown();
        return points[0];
    }

    @Override
    public String getName() {
        return "Nelder-Mead";
    }
}
