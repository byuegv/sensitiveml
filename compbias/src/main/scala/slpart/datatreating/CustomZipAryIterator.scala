package slpart.datatreating

import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import com.intel.analytics.bigdl.numeric.NumericFloat

class CustomZipAryIterator[T: ClassTag](iter: Iterator[(Array[ArrayBuffer[LabeledPoint]],Long)])
  extends Iterator[(ArrayBuffer[T],ArrayBuffer[T])]{
  override def hasNext: Boolean = {
    iter.hasNext
  }

  override def next(): (ArrayBuffer[T], ArrayBuffer[T]) = {
    val p = iter.next()

    val zipAndOrig = p._1
    val zipAry = zipAndOrig.head
    val origAry = zipAndOrig.last
    val compId = p._2 + 2

    val feaAry = new ArrayBuffer[Double]
    val labAry = new ArrayBuffer[Double]

    val compfea = zipAry(0).features.toArray
    //  val complab = zipAry(0).label + compId * compScale
    val complab = zipAry(0).label

    feaAry ++= compfea
    labAry += complab

    for(elem <- origAry){
      val origfea = elem.features.toArray
      val origlab = elem.label
      feaAry ++= origfea
      labAry += origlab
    }
    (feaAry.asInstanceOf[ArrayBuffer[T]],labAry.asInstanceOf[ArrayBuffer[T]])
  }
}