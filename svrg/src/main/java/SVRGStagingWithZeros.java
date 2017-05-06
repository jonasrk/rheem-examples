//package org.qcri.ml4all.GD.logical;
//
//import org.qcri.ml4all.abstraction.logical.LocalStage;
//import org.qcri.ml4all.api.ML4allContext;
//
///**
// * Created by zoi on 22/1/15.
// */
//
//public class SRVGStagingWithZeros extends LocalStage {
//
//    int features;
//
//    public SRVGStagingWithZeros(int features) {
//        this.features = features;
//    }
//
//    @Override
//    public void staging(ML4allContext context) {
//        double[] weights = new double[features];
//        double[] weightsBar = new double[features];
//        context.put("weights", weights);
//        context.put("weightsBar", weightsBar);
//        context.put("iter", 1);
//        context.put("step", 0.1);
//    }
//}
