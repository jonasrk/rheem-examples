package org.qcri.ml4all.GD.logical;


import org.qcri.ml4all.GD.gradients.Gradient;
import org.qcri.ml4all.abstraction.logical.Compute;
import org.qcri.ml4all.api.ML4allContext;

/**
 * Created by zoi on 8/3/16.
 */

public class SRVGCompute<R, V> extends Compute<R, V> {

    Gradient<R, double[], V> gradient;
    double[] sumGrad;
    double[] sumGradBar;
    int features;
    int m;

    public SRVGCompute(Gradient gradient, int features, int m) {
        this.gradient = gradient;
        sumGrad = new double[features + 1]; //position 0 is for the count
        sumGradBar = new double[features + 1]; //position 0 is for the count
        this.features = features;
        this.m = m;
    }

    public void setGradient (Gradient gradient, int features) {
        this.gradient = gradient;
        sumGrad = new double[features + 1]; //position 0 is for the count
        sumGradBar = new double[features + 1]; //position 0 is for the count
    }

    @Override
    public void initialise() {
        sumGrad = new double[features + 1];
        sumGradBar = new double[features + 1];
    }

    @Override
    public R process(V point, ML4allContext context) {
        int iteration = (int) context.getByKey("iter");
        double[] weights = (double[]) context.getByKey("weights");
        if ((iteration - 1) % m == 0) { //calculate only one gradient based on weights (because w and wBar are equal)
            return this.gradient.calculate(weights, point, sumGrad);
        }
        else { //calculate the gradient for the point using w and wBar
            sumGrad = (double []) this.gradient.calculate(weights, point, sumGrad);
            double[] weightsBar = (double[]) context.getByKey("weightsBar");
            sumGradBar = (double []) this.gradient.calculate(weightsBar, point, sumGradBar);
            double[] mergedGradients = mergeArrays(sumGrad, sumGradBar);
            return (R) mergedGradients;
        }
    }

    private static double[] mergeArrays(double[] a, double[] b) {
        int aLen = a.length;
        int bLen = b.length;
        double[] merged = new double[aLen + bLen];
        System.arraycopy(a, 0, merged, 0, aLen);
        System.arraycopy(b, 0, merged, aLen, bLen);
        return merged;
    }

}
