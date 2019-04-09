package slpart.datatreating

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


class CustomZipSampleIterator[T: ClassTag](iter: Iterator[(ArrayBuffer[T],ArrayBuffer[T])],singleFeatureLength: Int,comp2Orig: Int)
  extends Iterator[Sample[Float]]{
  override def hasNext: Boolean = {
    iter.hasNext
  }

  override def next(): Sample[Float] = {
    val p = iter.next()

    val feaAry = p._1
    val labAry = p._2
    val preSize = labAry.length
    val padFea = feaAry.slice(singleFeatureLength + 1, singleFeatureLength + singleFeatureLength + 1)
    val padLab = labAry(1)

    var curSize = preSize
    while (curSize < comp2Orig) {
      feaAry ++= padFea
      labAry += padLab
      curSize += 1
    }
    Sample(Tensor[Float](T(feaAry.head, feaAry.tail: _*)), Tensor[Float](T(labAry.head, labAry.tail: _*)))
  }
}
