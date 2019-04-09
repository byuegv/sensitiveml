package slpart.SLTransformer

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}


class SampleToLabeledPoint extends Transformer[Sample[Float],LabeledPoint]{
  override def apply(prev: Iterator[Sample[Float]]): Iterator[LabeledPoint] = {
    prev.map{sample =>{
      val shape = sample.feature().size()
      var prd = 1
      shape.foreach(x => prd = prd * x)
      val features = sample.feature().reshape(Array(1,prd)).squeeze().toArray().map(_.asInstanceOf[Double])
      val label = sample.label().squeeze().toArray()(0).asInstanceOf[Double]
      LabeledPoint(label,Vectors.dense(features))
    }}
  }

}

object SampleToLabeledPoint {
  def apply(): SampleToLabeledPoint= new SampleToLabeledPoint()

//  def main(args: Array[String]): Unit = {
//    val aryLabeled = Array(
//      LabeledPoint(1.0,Vectors.dense(1.0,2,0,4.0)),
//      LabeledPoint(2.0,Vectors.dense(3.0,4.0,3.3)),
//      LabeledPoint(3.0,Vectors.dense(9.0,8.7,3.1))
//    )
//    val samples = LabeledPointToSample()(aryLabeled.iterator)
//
//    val lab = SampleToLabeledPoint()(samples)
//
//    while(lab.hasNext){
//      val labx = lab.next()
//      System.out.println(s"features:\n${labx.features} \nlabel:${labx.label}")
//    }
//  }
}
