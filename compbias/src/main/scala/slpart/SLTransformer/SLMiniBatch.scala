package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SLMiniBatch[T: ClassTag](slsamples: Array[(Sample[T],Array[Sample[T]])])
   (implicit ev: TensorNumeric[T]) extends MiniBatch[T]{
  val firstSample = slsamples.head._1
  val cMiniBatch = MiniBatch(firstSample.numFeature(),firstSample.numLabel()).set(slsamples.map(_._1))

  val oMiniBatch = MiniBatch(firstSample.numFeature(),firstSample.numLabel())
    .set(slsamples.map(_._2).reduce(_ ++ _))

  def getOrigInput(): Activity = {
    oMiniBatch.getInput()
  }
  def getOrigTarget(): Activity = {
    oMiniBatch.getTarget()
  }
  def origSize(): Int = {
    oMiniBatch.size()
  }

  override def getInput(): Activity = {
    cMiniBatch.getInput()
  }

  override def getTarget(): Activity = {
    cMiniBatch.getTarget()
  }

  override def size(): Int = {
    cMiniBatch.size()
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    new SLMiniBatch[T](slsamples.slice(offset-1,length))
  }

  override def set(slsamples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): SLMiniBatch.this.type = {
    this
  }
}
