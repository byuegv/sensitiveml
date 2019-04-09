package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.Iterator
import scala.reflect.ClassTag

class SamplesToSLMiniBatch[T: ClassTag](
                             totalBatch: Int,
                             partitionNum: Option[Int] = None)
                                       (implicit ev: TensorNumeric[T]) extends Transformer[(Sample[T],Array[Sample[T]]), MiniBatch[T]] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)
  private val batchSize = batchPerPartition
  private val sampleData = new Array[(Sample[T],Array[Sample[T]])](batchSize)
  override def apply(prev: Iterator[(Sample[T],Array[Sample[T]])]): Iterator[MiniBatch[T]] = {
    new Iterator[MiniBatch[T]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            sampleData(i) = sample
            i += 1
          }

          if (i < batchSize) {
            new SLMiniBatch[T](sampleData.slice(0, i))
          } else {
            new SLMiniBatch[T](sampleData)
          }
        } else {
          null
        }
      }
    }
  }
}

object SamplesToSLMiniBatch{
  def apply[T: ClassTag](
             totalBatch: Int,
             partitionNum: Option[Int] = None)
           (implicit ev: TensorNumeric[T]): SamplesToSLMiniBatch[T] =
    new SamplesToSLMiniBatch[T](totalBatch,partitionNum)
}