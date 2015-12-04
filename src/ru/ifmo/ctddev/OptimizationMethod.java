package ru.ifmo.ctddev;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * An abstract class for minimizing a non-smooth function.
 * The function is considered to be hard computable, so minimal number
 * of function calls is desired.
 */
public abstract class OptimizationMethod {

    /**
     * Represents a point in the search space with the value of the function.
     */
    public static class Point implements Comparable<Point> {
        /**
         * Point in the search space.
         */
        public final double[] x;

        /**
         * Value of the function in this point.
         */
        public Double quality;

        public Point(double[] x) {
            this.x = Arrays.copyOf(x, x.length);
        }

        @Override
        public int compareTo(Point o) {
            if (this.quality < o.quality) {
                return -1;
            } else if (this.quality > o.quality) {
                return 1;
            } else {
                return 0;
            }
        }
    }


    /**
     * The result of optimization of some function be some optimization method.
     */
    public static class OptimizationResult {
        /**
         * The best point suggested by the optimization method.
         */
        public Point result;

        /**
         * The list of calls of the optimized function.
         */
        public List<Point> log;

        protected OptimizationResult(Point result, EvaluatorProxy proxy) {
            this.result = result;
            this.log = proxy.callsLog;
        }
    }

    /**
     * Auxiliary class for storing log of function calls.
     */
    protected static class EvaluatorProxy implements Function<double[], Double> {
        /**
         * The function to optimize.
         */
        private final Function<double[], Double> function;

        /**
         * The log of the function calls, contains points and function values in them.
         */
        private List<Point> callsLog = new ArrayList<>();

        /**
         * The meaning of "optimization", minimizes the function if set to true, maximizes it otherwise
         */
        private boolean minimize;

        public EvaluatorProxy(Function<double[], Double> function, boolean minimize) {
            this.function = function;
            this.minimize = minimize;
        }

        @Override
        public Double apply(double[] doubles) {
            double y = function.apply(doubles);
            synchronized (this) {
                Point p = new Point(Arrays.copyOf(doubles, doubles.length));
                p.quality = y;
                callsLog.add(p);
            }
            if (minimize) {
                return y;
            } else {
                return -y;
            }
        }
    }

    /**
     * Returns the (local) minimum {@link ru.ifmo.ctddev.OptimizationMethod.Point} of the function.
     * @param evaluator the function to minimize
     * @param arity the arity of the function
     * @param out the stream to output info
     * @return the (local) minimum point
     */
    protected abstract Point minimize(Function<double[], Double> evaluator, int arity, PrintStream out);

    /**
     * Returns the method's name.
     * @return {@link String} - the name.
     */
    public abstract String getName();

    /**
     * Returns the {@link ru.ifmo.ctddev.OptimizationMethod.OptimizationResult} of the function.
     * @param evaluator the function to optimize
     * @param arity the arity of the function
     * @param out the stream to output info
     * @param minimize the meaning of "optimization", minimizes the function if set to true, maximizes it otherwise
     * @return the result of the optimization
     */
    public OptimizationResult optimize(Function<double[], Double> evaluator,
            int arity, PrintStream out, boolean minimize) {
        EvaluatorProxy evaluatorProxy = new EvaluatorProxy(evaluator, minimize);
        Point point = minimize(evaluatorProxy, arity, out);
        if (!minimize) {
            point.quality = -point.quality;
        }
        return new OptimizationResult(point, evaluatorProxy);
    }

    /**
     * Returns the {@link ru.ifmo.ctddev.OptimizationMethod.OptimizationResult} of the function.
     * @param evaluator function to optimize
     * @param arity arity of the function
     * @param minimize meaning of "optimization", minimizes the function if set to true, maximizes it otherwise
     * @return the result of the optimization
     */
    public OptimizationResult optimize(Function<double[], Double> evaluator, int arity, boolean minimize) {
        return optimize(evaluator, arity, System.out, minimize);
    }
}
