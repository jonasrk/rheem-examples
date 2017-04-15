//package org.qcri.ml4all.GD.logical;
//
//
//import org.qcri.ml4all.abstraction.logical.SampleSize;
//import org.qcri.ml4all.api.ML4allContext;
//
///**
// * Created by zoi on 8/3/15.
// */
//public class SRVGDSampleSize extends SampleSize {
//
//    long totalSize;
//    long batchSize;
//    int m; // the frequency to use batch GD, every m iterations don't sample
//
//    public SRVGDSampleSize(long totalSize, long batchSize, int m) {
//        this.totalSize = totalSize;
//        this.batchSize = batchSize;
//        this.m = m;
//    }
//
//    @Override
//    public long getSampleSize(ML4allContext context) {
//        int iteration = (int) context.getByKey("iter");
//        System.out.println("iteration:" + iteration);
//        if ((iteration - 1) % m == 0) {
//            return totalSize;
//        }
//        else
//            return batchSize;
//    }
//}
