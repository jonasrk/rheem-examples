//package org.qcri.ml4all.GD.logical;
//
//
//
//import org.qcri.ml4all.abstraction.logical.UpdateLocal;
//import org.qcri.ml4all.api.ML4allContext;
//
//import java.util.Arrays;
//
///**
// * Created by zoi on 22/1/15.
// */
//
//public class SRVGUpdate extends UpdateLocal< double[], double[]> {
//
//    int m;
//    double lamda = 0;
//
//    public SRVGUpdate(int m) { this.m = m; }
//
//    public SRVGUpdate(int m, double lamda) {
//        this.m = m;
//        this.lamda = lamda;
//    }
//
//
//    @Override
//    public double[] process(double[] grad, ML4allContext context) {
//
//        double[] weights = (double[]) context.getByKey("weights");
//        double stepSize = (double) context.getByKey("step");
//        int iteration = (int) context.getByKey("iter");
//
//        double alpha = (stepSize / iteration);
//        double newValue;
//        if ((iteration % m) - 1 == 0) { //every m iterations, store the sum of gradients and use BGD update rule
//            if (iteration > 1) {
//                double[] weightsBar = Arrays.copyOf(weights, weights.length);
//                context.put("weightsBar", weightsBar);
//            }
//            double count = grad[0];
//            context.put("mu", grad);
//            double[] newWeights = new double[weights.length];
//            for (int j = 0; j < weights.length; j++) {
//                newValue = weights[j] - alpha * (1.0/count) * grad[j + 1] + lamda*alpha*weights[j];
//                newWeights[j] = newValue;
//            }
//            return newWeights;
//        }
//        else {
//            double[] mu = (double[]) context.getByKey("mu");
//            double count = mu[0];
//
//            double[] sumGrad = new double[weights.length+1];
//            double[] sumGradBar = new double[weights.length+1];
//            System.arraycopy(grad, 0, sumGrad, 0, weights.length);
//            System.arraycopy(grad, weights.length+1, sumGradBar, 0, weights.length);
////            System.out.println("got: sumGrad:" + Arrays.toString(sumGrad));
////            System.out.println("got: sumGrad.size:" + sumGrad.length);
////            System.out.println("got: sumGradBar:" + Arrays.toString(sumGradBar));
////            System.out.println("got: sumGradBar.size:" + sumGradBar.length);
//
//            System.out.println("count:" + count);
//
//            double[] newWeights = new double[weights.length];
//            for (int j = 0; j < weights.length; j++) {
//                newValue = weights[j] - alpha * (grad[j + 1] - grad[weights.length + j + 2] + (1.0/count) * mu[j]) + lamda*alpha*weights[j];
//                newWeights[j] = newValue;
//            }
//            return newWeights;
//        }
//    }
//
//    @Override
//    public ML4allContext assign(double[] input, ML4allContext context) {
//        System.out.println("assign");
//        context.put("weights", input);
//        int iteration = (int) context.getByKey("iter");
//        context.put("iter", ++iteration);
//        return context;
//    }
//}
