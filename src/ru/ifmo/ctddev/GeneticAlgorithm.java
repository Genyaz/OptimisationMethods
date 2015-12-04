package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

/**
 * The optimization method based on the natural selection.
 * Current realization is written according to Wikipedia article
 *
 * https://en.wikipedia.org/wiki/Genetic_algorithm
 */
public class GeneticAlgorithm extends OptimizationMethod {

    protected final double[][] boundaries;
    protected final double mutationCoef, mutationRate, selection, elite, diff;
    protected final int populationSize;
    protected final Random r;

    /**
     * Constructs a new genetic algorithm.
     * @param boundaries boundaries of the search space
     * @param populationSize size of the population
     * @param mutationCoef maximum random addition/subtraction to coordinate
     * @param mutationRate chance of position's mutation
     * @param selection fraction of the best of the population allowed for breeding
     * @param elite fraction of the best of the population translated to the next generation
     * @param diff stopping-criterion - the method stops after the difference between the best and
     *             the worst solution is less than diff
     */
    public GeneticAlgorithm(double[][] boundaries, int populationSize,
                            double mutationCoef, double mutationRate, double selection, double elite, double diff) {
        this.boundaries = boundaries;
        this.populationSize = populationSize;
        this.mutationCoef = mutationCoef;
        this.mutationRate = mutationRate;
        this.selection = selection;
        this.elite = elite;
        this.diff = diff;
        this.r = new Random(System.currentTimeMillis());
    }

    /**
     * Default genetic algorithm constructor.
     */
    public GeneticAlgorithm() {
        this(new double[][]{{-8, 8}, {-8, 8}}, 20, 0.1, 0.1, 0.5, 0.1, 0.1);
    }

    /**
     * Chooses parent from the breed with the probability proportional to
     * the difference of quality between this individual and the worst.
     * @param breeding breeding individuals
     * @return chosen parent
     */
    protected Point chooseParent(Point[] breeding) {
        double maxQuality = Double.MIN_VALUE;
        for (Point p : breeding) {
            maxQuality = Math.max(maxQuality, p.quality);
        }
        double sumQuality = 0;
        for (Point p: breeding) {
            sumQuality += (maxQuality - p.quality);
        }
        double q = r.nextDouble() * sumQuality;
        int j = 0;
        sumQuality = 0;
        while (j < breeding.length - 1 && sumQuality + (maxQuality - breeding[j].quality) < q) {
            sumQuality += (maxQuality - breeding[j].quality);
            j++;
        }
        return breeding[j];
    }

    /**
     * Crossovers two chromosomes with two-point crossover.
     * @param x first parent
     * @param y second parent
     * @return crossovered chromosome
     */
    protected double[] crossover(double[] x, double[] y) {
        int p1 = (int)(r.nextDouble() * x.length);
        int p2 = (int)(r.nextDouble() * x.length);
        if (p1 > p2) {
            int tmp = p1;
            p1 = p2;
            p2 = tmp;
        }
        double[] result = new double[x.length];
        for (int i = 0; i < p1; i++) {
            result[i] = x[i];
        }
        for (int i = p1; i < p2; i++) {
            result[i] = y[i];
        }
        for (int i = p2; i < x.length; i++) {
            result[i] = x[i];
        }
        return result;
    }

    /**
     * Mutates given chromosome with a chance of random addition to every coordinate.
     * @param x chromosome
     * @param mutationRate chance of position's mutation
     * @param mutationCoef maximum random addition/subtraction to coordinate
     * @return mutated chromosome
     */
    protected double[] mutate(double[] x, double mutationRate, double mutationCoef) {
        for (int i = 0; i < x.length; i++) {
            if (r.nextDouble() < mutationRate) {
                x[i] += mutationCoef * 2 * (r.nextDouble() - 1);
            }
        }
        return x;
    }

    @Override
    protected Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out) {
        Point[] population = new Point[populationSize];
        for (int i = 0; i < populationSize; i++) {
            double[] x = new double[arity];
            for (int j = 0; j < arity; j++) {
                x[j] = (boundaries[j][1] - boundaries[j][0]) * r.nextDouble() + boundaries[j][0];
            }
            population[i] = new Point(x);
            population[i].quality = evaluator.apply(x);
        }
        final int eliteSize = (int)(elite * populationSize);
        final int selectionSize = (int)(selection * populationSize);
        while (true) {
            Arrays.sort(population);
            //out.println(population[0].quality + " " + population[populationSize - 1].quality);
            if (population[populationSize - 1].quality - population[0].quality < diff) {
                return population[0];
            }
            Point[] breeding = Arrays.copyOfRange(population, 0, selectionSize);
            Point[] nextGen = new Point[populationSize];
            for (int i = 0; i < eliteSize; i++) {
                nextGen[i] = population[i];
            }
            for (int i = 0; i < populationSize - eliteSize; i++) {
                Point parent1 = chooseParent(breeding), parent2 = chooseParent(breeding);
                double[] x = mutate(crossover(parent1.x, parent2.x), mutationRate, mutationCoef);
                nextGen[eliteSize + i] = new Point(x);
                nextGen[eliteSize + i].quality = evaluator.apply(x);
            }
            population = nextGen;
        }
    }

    @Override
    public String getName() {
        return "Genetic algorithm";
    }
}
