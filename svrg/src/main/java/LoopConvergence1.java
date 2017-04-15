//package org.qcri.ml4all.GD.logical;
//
//
//import org.qcri.ml4all.abstraction.logical.Loop;
//import org.qcri.ml4all.api.ML4allContext;
//import org.qcri.rheem.basic.data.Tuple2;
//
///**
// * Created by zoi on 1/2/15.
// */
//public class LoopConvergence1 extends Loop<Tuple2<Double, Double>, double[]> {
//
//    double accuracy;
//
//    public LoopConvergence1(double accuracy) {
//        this.accuracy = accuracy;
//    }
//
//    @Override
//    public Tuple2<Double, Double> prepareConvergenceDataset(double[] input, ML4allContext context) {
//        double[] weights = (double[]) context.getByKey("weights");
//        double normDiff = 0.0;
//        double normWeights = 0.0;
//        for (int j = 0; j < weights.length; j++) {
//            normDiff += Math.abs(weights[j] - input[j]);
//            normWeights += Math.abs(input[j]);
//        }
//        return new Tuple2(normDiff, normWeights);
//    }
//
//    @Override
//    public boolean terminate(Tuple2<Double, Double> input) {
//        return input.field0 < accuracy * Math.max(input.field1, 1.0);
//    }
//}
