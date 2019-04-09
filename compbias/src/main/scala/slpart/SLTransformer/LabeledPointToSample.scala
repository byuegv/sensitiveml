package slpart.SLTransformer

import org.apache.spark.mllib.regression.LabeledPoint
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor


class LabeledPointToSample extends Transformer[LabeledPoint,Sample[Float]]{
  override def apply(prev: Iterator[LabeledPoint]): Iterator[Sample[Float]] = {
    prev.map(labeledPoint =>{
      val features = labeledPoint.features.toArray
      val label = labeledPoint.label
      Sample(Tensor(T(features.head,features.tail: _*)),Tensor(T(label)))
    })
  }
}

object LabeledPointToSample{
  def apply(): LabeledPointToSample = new LabeledPointToSample()
}
