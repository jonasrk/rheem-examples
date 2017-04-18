import org.qcri.rheem.api.DataQuantaBuilder;
import org.qcri.rheem.api.JavaPlanBuilder;
import org.qcri.rheem.basic.data.Tuple2;
import org.qcri.rheem.core.api.RheemContext;
import org.qcri.rheem.core.function.ExecutionContext;
import org.qcri.rheem.core.function.FunctionDescriptor;
import org.qcri.rheem.core.util.RheemCollections;
import org.qcri.rheem.core.util.Tuple;
import org.qcri.rheem.java.Java;
import org.qcri.rheem.spark.Spark;

import java.io.File;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;


/**
 * This class executes a stochastic gradient descent optimization on Rheem.
 */
public class RunSVRG {

    // Default parameters.
    static String relativePath = "svrg/src/main/resources/adult.zeros";
    static int datasetSize  = 100827;
    static int features = 123;

    //these are for SVRG/mini run to convergence
    static int sampleSize = 10;
    static double accuracy = 0.001;
    static int max_iterations = 1000;


    public static void main (String... args) throws MalformedURLException {

        //Usage: <data_file> <#features> <sparse> <binary>
        if (args.length > 0) {
            relativePath = args[0];
            datasetSize = Integer.parseInt(args[1]);
            features = Integer.parseInt(args[2]);
            max_iterations = Integer.parseInt(args[3]);
            accuracy = Double.parseDouble(args[4]);
            sampleSize = Integer.parseInt(args[5]);
        }
        else {
            System.out.println("Usage: java <main class> [<dataset path> <dataset size> <#features> <max iterations> <accuracy> <sample size>]");
            System.out.println("Loading default values");
        }

        String file = new File(relativePath).getAbsoluteFile().toURI().toURL().toString();

        System.out.println("max #iterations:" + max_iterations);
        System.out.println("accuracy:" + accuracy);

        new RunSVRG().execute(file, features);
    }


    public void execute(String fileName, int features) {
        RheemContext rheemContext = new RheemContext().with(Java.basicPlugin()).with(Spark.basicPlugin());
        JavaPlanBuilder javaPlanBuilder = new JavaPlanBuilder(rheemContext);

        List<double[]> weights = Arrays.asList(new double[features]);
        final DataQuantaBuilder<?, double[]> weightsBuilder = javaPlanBuilder
                .loadCollection(weights)
                .withTargetPlatform(Java.platform())
                .withName("init weights");

        final DataQuantaBuilder<?, double[]> transformBuilder = javaPlanBuilder
                .readTextFile(fileName).withName("source")
                .withTargetPlatform(Java.platform())
                .map(new Transform(features)).withName("transform")
                .withTargetPlatform(Java.platform());


        // START OF OLD LOOP

//        Collection<double[]> results  =
//                weightsBuilder.doWhile(new LoopCondition(accuracy, max_iterations), w -> {
//
//                    DataQuantaBuilder<?, double[]> newWeightsDataset = transformBuilder
//                            .sample(sampleSize)
////                    .<double[]>customOperator(new SparkRandomPartitionSampleOperator<>(sampleSize, datasetSize, DataSetType.createDefault(double[].class)))
////                    .withOutputClass(double[].class)
//                            .map(new ComputeLogisticGradient()).withBroadcast(w, "weights").withName("compute")
//                            .reduce(new Sum()).withName("reduce")
//                            .map(new WeightsUpdate()).withBroadcast(w, "weights").withName("update");
//
//                    DataQuantaBuilder<?, Tuple2<Double, Double>> convergenceDataset = newWeightsDataset.map(new ComputeNorm()).withBroadcast(w, "weights");
//
//                    return new Tuple<>(newWeightsDataset, convergenceDataset);
//                }).collect();

        // END OF OLD LOOP


        // START OF NEW, UNROLLED LOOP

        // START iteration ZERO

        List<Integer> current_iteration = Arrays.asList(0);

        DataQuantaBuilder<?, Integer> iteration_list = javaPlanBuilder
                .loadCollection(current_iteration)
                .withTargetPlatform(Java.platform());


        // operator lists:
        ArrayList<DataQuantaBuilder<?, double[]>> FullOperatorList = new ArrayList<DataQuantaBuilder<?, double[]>>();
        ArrayList<DataQuantaBuilder<?, double[]>> muOperatorList = new ArrayList<DataQuantaBuilder<?, double[]>>();
        ArrayList<DataQuantaBuilder<?, double[]>> PartialOperatorList = new ArrayList<DataQuantaBuilder<?, double[]>>();

        PartialOperatorList.add(
                transformBuilder
//                        .sample(sampleSize)
                        .map(new ComputeLogisticGradientFullIteration())
                        .withTargetPlatform(Java.platform())
                        .withBroadcast(weightsBuilder, "weights")
                        .withName("compute")
                        .reduce(new Sum()).withName("reduce")
                        .withTargetPlatform(Java.platform())
                        .map(new WeightsUpdateFullIteration())
                        .withTargetPlatform(Java.platform())
                        .withBroadcast(weightsBuilder, "weights")
                        .withBroadcast(iteration_list, "current_iteration")
                        .withName("update")
        );




        // END iteration ZERO

        // START other iterations

        int iterations = 125; // TODO JRK move to parameters

        for (int i = 1; i < iterations; i++) {

            if (i % 2 == 1){

                current_iteration = Arrays.asList(i);

                iteration_list = javaPlanBuilder
                        .loadCollection(current_iteration)
                        .withTargetPlatform(Java.platform());

                muOperatorList.add(transformBuilder
//                    .sample(sampleSize)
//                    .withTargetPlatform(Java.platform())
                        .map(new ComputeLogisticGradientFullIteration())
                        .withTargetPlatform(Java.platform())
                        .withBroadcast(PartialOperatorList.get(PartialOperatorList.size() - 1), "weights")
                        .withName("compute")
                        .reduce(new Sum()).withName("reduce")
                        .withTargetPlatform(Java.platform()));

                FullOperatorList.add(muOperatorList.get(muOperatorList.size() - 1)
                        .map(new WeightsUpdateFullIteration())
                        .withTargetPlatform(Java.platform())
                        .withBroadcast(PartialOperatorList.get(PartialOperatorList.size() - 1), "weights")
                        .withBroadcast(iteration_list, "current_iteration")
                        .withName("update"));
            } else {

                current_iteration = Arrays.asList(i);

                iteration_list = javaPlanBuilder
                        .loadCollection(current_iteration)
                        .withTargetPlatform(Java.platform());

                PartialOperatorList.add(transformBuilder
                        .sample(1)
                        .withTargetPlatform(Java.platform())
                        .map(new ComputeLogisticGradient())
                        .withTargetPlatform(Java.platform())
                        .withBroadcast(FullOperatorList.get(FullOperatorList.size() - 1), "weights")
                        .withName("compute")
                        .map(new WeightsUpdate())
                        .withTargetPlatform(Java.platform())
                        .withBroadcast(muOperatorList.get(muOperatorList.size() - 1), "mu")
                        .withBroadcast(FullOperatorList.get(FullOperatorList.size() - 1), "weights")
                        .withBroadcast(FullOperatorList.get(FullOperatorList.size() - 1), "weightsBar")
                        // TODO JRK all these Full Operator references can not be right
                        .withBroadcast(iteration_list, "current_iteration")
                        .withName("update"));
            }
        }
        // END other iterations

        // END OF NEW LOOP

        System.out.println("Output weights:" + Arrays.toString(RheemCollections.getSingle(FullOperatorList.get(FullOperatorList.size() - 1).collect())));

//        System.out.println("Output weights:" + Arrays.toString(RheemCollections.getSingle(results)));

    }
}

class Transform implements FunctionDescriptor.SerializableFunction<String, double[]> {

    int features;

    public Transform (int features) {
        this.features = features;
    }

    @Override
    public double[] apply(String line) {
        String[] pointStr = line.split(" ");
        double[] point = new double[features+1];
        point[0] = Double.parseDouble(pointStr[0]);
        for (int i = 1; i < pointStr.length; i++) {
            if (pointStr[i].equals("")) {
                continue;
            }
            String kv[] = pointStr[i].split(":", 2);
            point[Integer.parseInt(kv[0])-1] = Double.parseDouble(kv[1]);
        }
        return point;
    }
}

class ComputeLogisticGradient implements FunctionDescriptor.ExtendedSerializableFunction<double[], double[]> {

    double[] weights;

    @Override
    public double[] apply(double[] point) {
        double[] gradient = new double[point.length];
        double dot = 0;
        for (int j = 0; j < weights.length; j++)
            dot += weights[j] * point[j + 1];

        for (int j = 0; j < weights.length; j++)
            gradient[j + 1] = ((1 / (1 + Math.exp(-1 * dot))) - point[0]) * point[j + 1];

        gradient[0] = 1; //counter for the step size required in the update
//        System.out.println("half " + gradient[0]);
        return gradient;
    }

    @Override
    public void open(ExecutionContext executionContext) {
        this.weights = (double[]) executionContext.getBroadcast("weights").iterator().next();
    }
}

class ComputeLogisticGradientFullIteration implements FunctionDescriptor.ExtendedSerializableFunction<double[], double[]> {

    double[] weights, weightsBar;

    double[] calculateGradient(double[] weights, double[] point){
        double[] gradient = new double[point.length];
        double dot = 0;
        for (int j = 0; j < weights.length; j++)
            dot += weights[j] * point[j + 1];

        for (int j = 0; j < weights.length; j++)
            gradient[j + 1] = ((1 / (1 + Math.exp(-1 * dot))) - point[0]) * point[j + 1];

        gradient[0] = 1; //counter for the step size required in the update
        return gradient;
    }

    @Override
    public double[] apply(double[] point) {
        double[] sumGrad = (double []) calculateGradient(weights, point);
        double[] sumGradBar = (double []) calculateGradient(weightsBar, point);
        double[] mergedGradients = mergeArrays(sumGrad, sumGradBar);
        return mergedGradients;
    }

    private static double[] mergeArrays(double[] a, double[] b) {
        int aLen = a.length;
        int bLen = b.length;
        double[] merged = new double[aLen + bLen];
        System.arraycopy(a, 0, merged, 0, aLen);
        System.arraycopy(b, 0, merged, aLen, bLen);
        return merged;
    }

    @Override
    public void open(ExecutionContext executionContext) {
        this.weights = (double[]) executionContext.getBroadcast("weights").iterator().next();
        this.weightsBar = (double[]) executionContext.getBroadcast("weightsBar").iterator().next();
    }
}

class Sum implements FunctionDescriptor.SerializableBinaryOperator<double[]> {

    @Override
    public double[] apply(double[] o, double[] o2) {
        double[] g1 = o;
        double[] g2 = o2;

        if (g2 == null) //samples came from one partition only
            return g1;

        if (g1 == null) //samples came from one partition only
            return g2;

        double[] sum = new double[g1.length];
        sum[0] = g1[0] + g2[0]; //count
        for (int i = 1; i < g1.length; i++)
            sum[i] = g1[i] + g2[i];

        return sum;
    }
}

class WeightsUpdate implements FunctionDescriptor.ExtendedSerializableFunction<double[], double[]> {

    double[] weights;
    double[] mu;
    int current_iteration;
    double lambda = 0;

    double stepSize = 1;
    double regulizer = 0;

    public WeightsUpdate () { }

    public WeightsUpdate (double stepSize, double regulizer) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
    }

    @Override
    public double[] apply(double[] input) {

        double count = input[0];
        double alpha = (stepSize / (current_iteration+1));
        System.out.println("### current_iteration: " + current_iteration);

        double[] newWeights = new double[weights.length];
        for (int j = 0; j < weights.length; j++) {
            double regulizer_term = (1 - alpha * regulizer); // TODO JRK ignore this for now
            double old_weight_term = weights[j];
            double step_size_term = alpha * (1.0 / count);
            double gradient_term = input[j + 1];
            newWeights[j] = regulizer_term * old_weight_term - step_size_term * gradient_term;
            double svrg_gradient_term =  (input[j + 1] - input[weights.length + j + 2] + (1.0/count) * mu[j]);
            double svrg_regulizer_term = lambda*alpha*weights[j]; // TODO JRK is it really?
            double svrg_stepsize_term = alpha;
            newWeights[j] = old_weight_term - svrg_stepsize_term * svrg_gradient_term + svrg_regulizer_term;
        }
        return newWeights;
    }

    @Override
    public void open(ExecutionContext executionContext) {
        this.weights = (double[]) executionContext.getBroadcast("weights").iterator().next();
        this.mu = (double[]) executionContext.getBroadcast("mu").iterator().next();
        this.current_iteration = ((Integer) executionContext.getBroadcast("current_iteration").iterator().next());
    }
}

class WeightsUpdateFullIteration implements FunctionDescriptor.ExtendedSerializableFunction<double[], double[]> {

    double[] weights;
    int current_iteration;

    double stepSize = 1;
    double regulizer = 0;

    public WeightsUpdateFullIteration () { }

    public WeightsUpdateFullIteration (double stepSize, double regulizer) {
        this.stepSize = stepSize;
        this.regulizer = regulizer;
    }

    @Override
    public double[] apply(double[] input) {

//        System.out.println("### in WeightsUpdate function");
//        System.out.println("### input[0]: " + input[0]);
//        System.out.println("### weights.length: " + weights.length);

        double count = input[0];
        double alpha = (stepSize / (current_iteration+1));
//        System.out.println("### alpha: " + alpha);
//        System.out.println("### stepSize: " + stepSize);
        System.out.println("### current_iteration: " + current_iteration);

        double[] newWeights = new double[weights.length];
        for (int j = 0; j < weights.length; j++) {
//            System.out.println("### j: " + j);
//            System.out.println("### regulizer: " + regulizer);
//            System.out.println("### weights[j]: " + weights[j]);
//            System.out.println("### count: " + count);
//            System.out.println("### input[j + 1]: " + input[j + 1]);
            newWeights[j] = (1 - alpha * regulizer) * weights[j] - alpha * (1.0 / count) * input[j + 1];
//            System.out.println("### newWeights[j]: " + newWeights[j]);
        }
//        System.out.println(newWeights);
        return newWeights;
    }

    @Override
    public void open(ExecutionContext executionContext) {
        this.weights = (double[]) executionContext.getBroadcast("weights").iterator().next();
        this.current_iteration = ((Integer) executionContext.getBroadcast("current_iteration").iterator().next());
    }
}

class ComputeNorm implements FunctionDescriptor.ExtendedSerializableFunction<double[], Tuple2<Double, Double>> {

    double[] previousWeights;

    @Override
    public Tuple2<Double, Double> apply(double[] weights) {
        double normDiff = 0.0;
        double normWeights = 0.0;
        for (int j = 0; j < weights.length; j++) {
//            normDiff += Math.sqrt(Math.pow(Math.abs(weights[j] - input[j]), 2));
            normDiff += Math.abs(weights[j] - previousWeights[j]);
//            normWeights += Math.sqrt(Math.pow(Math.abs(input[j]), 2));
            normWeights += Math.abs(weights[j]);
        }
        return new Tuple2(normDiff, normWeights);
    }

    @Override
    public void open(ExecutionContext executionContext) {
        this.previousWeights = (double[]) executionContext.getBroadcast("weights").iterator().next();
    }
}

class LoopCondition implements FunctionDescriptor.ExtendedSerializablePredicate<Collection<Tuple2<Double, Double>>> {

    public double accuracy;
    public int max_iterations;

    private int current_iteration;

    public LoopCondition(double accuracy, int max_iterations) {
        this.accuracy = accuracy;
        this.max_iterations = max_iterations;
    }

    @Override
    public boolean test(Collection<Tuple2<Double, Double>> collection) {
        Tuple2<Double, Double> input = RheemCollections.getSingle(collection);
        System.out.println("Running iteration: " + current_iteration);
        return (input.field0 < accuracy * Math.max(input.field1, 1.0) || current_iteration > max_iterations);
    }

    @Override
    public void open(ExecutionContext executionContext) {
        this.current_iteration = executionContext.getCurrentIteration();
    }
}