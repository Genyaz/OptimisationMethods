package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.function.Function;

/**
 * The optimization method based on the natural selection with Lamarck ideas .
 * Current realization is written according to Wikipedia article
 *
 * https://en.wikipedia.org/wiki/Genetic_algorithm
 */
public class MemeticAlgorithm extends GeneticAlgorithm {

    protected final double indOptProb, indOptStep;
    protected final int indOptIter;

    /**
     * Constructs a new memetic algorithm. Individual optimization is local search with fixed step.
     * @param boundaries boundaries of the search space
     * @param populationSize size of the population
     * @param mutationCoef maximum random addition/subtraction to coordinate
     * @param mutationRate chance of position's mutation
     * @param selection fraction of the best of the population allowed for breeding
     * @param elite fraction of the best of the population translated to the next generation
     * @param diff stopping-criterion - the method stops after the difference between the best and
     *             the worst solution is less than diff
     * @param indOptProb probability of individual optimization
     * @param indOptIter number of inidividual optimization iterations
     * @param indOptStep size of individual optimization step
     */
    public MemeticAlgorithm(double[][] boundaries, int populationSize,
            double mutationCoef, double mutationRate, double selection,
            double elite, double diff, double indOptProb, int indOptIter,
            double indOptStep) {
        super(boundaries, populationSize, mutationCoef, mutationRate, selection, elite, diff);
        this.indOptIter = indOptIter;
        this.indOptProb = indOptProb;
        this.indOptStep = indOptStep;
    }

    /**
     * Default memetic algorithm constructor.
     */
    public MemeticAlgorithm() {
        this(new double[][]{{-8, 8}, {-8, 8}}, 20, 0.1, 0.1, 0.5, 0.1, 0.1, 0.05, 2, 0.05);
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
            for (int i = 0; i < populationSize; i++) {
                if (r.nextDouble() < indOptProb) {
                    Point best = population[i];
                    for (int t = 0; t < indOptIter; t++) {
                        for (int j = 0; j < arity; j++) {
                            best.x[j] += indOptStep;
                            Point plusStep = new Point(best.x);
                            plusStep.quality = evaluator.apply(plusStep.x);
                            best.x[j] -= 2 * indOptStep;
                            Point minusStep = new Point(best.x);
                            minusStep.quality = evaluator.apply(minusStep.x);
                            best.x[j] += indOptStep;
                            if (plusStep.quality < best.quality) {
                                best = plusStep;
                            }
                            if (minusStep.quality < best.quality) {
                                best = minusStep;
                            }
                        }
                    }
                    population[i] = best;
                }
            }
        }
    }

    @Override
    public String getName() {
        return "Memetic algorithm";
    }
}
