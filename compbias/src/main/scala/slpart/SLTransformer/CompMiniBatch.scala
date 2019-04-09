package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._

import scala.reflect.ClassTag

class CompMiniBatch[T: ClassTag](
    override val inputData: Array[Tensor[T]],
    override  val targetData: Array[Tensor[T]],
    featurePaddingParam: Option[PaddingParam[T]] = None,
    labelPaddingParam: Option[PaddingParam[T]] = None,
    val comp: Int = -1)(implicit ev: TensorNumeric[T]) extends ArrayTensorMiniBatch [T](
  inputData, targetData, featurePaddingParam, labelPaddingParam
){
  require(inputData.length > 0 && comp > 0 ,
  s"Input data in CompMiniBatch is empty, comp -> orig should > 0 get comp = ${comp}")

//  override protected  var batchSize = 0
//  override protected var unlabeled = false

  override val (featurePadding, featurePaddingStrategy) = if (featurePaddingParam.isDefined) {
    (featurePaddingParam.get.paddingTensor, featurePaddingParam.get.paddingStrategy)
  } else {
    (None, new DefaultPadding)
  }

  override val (labelPadding, labelPaddingStrategy) = if (labelPaddingParam.isDefined) {
    (labelPaddingParam.get.paddingTensor, labelPaddingParam.get.paddingStrategy)
  } else {
    (None, new DefaultPadding)
  }

  private val oMiniBatch = getOrig()
  private val input: Activity = oMiniBatch.getInput()
  private val target: Activity = oMiniBatch.getTarget()

  private val cMiniBatch = getComp()
  private val compInput: Activity = cMiniBatch.getInput()
  private val compTarget: Activity = cMiniBatch.getTarget()

  def getCompInput: Activity = {
    compInput
  }
  def getCompTarget: Activity = {
    compTarget
  }

  override def size(): Int = {
    if (inputData.head.nElement() == 0) {
      0
    } else {
      inputData.head.size(1)
    }
  }

  def getComp(): MiniBatch[T] = {
    val inputs = new Array[Tensor[T]](inputData.length)
    val targets = new Array[Tensor[T]](targetData.length)
    val allInBatch = inputData.head.size(1)

    val compFeatureSize = inputData.head.size()
    compFeatureSize(0) = allInBatch / comp
    val compTargetSize = targetData.head.size()
    compTargetSize(0) = allInBatch / comp
    for(u <- 0 until inputs.length) inputs(u) = Tensor[T](compFeatureSize)
    for(u <- 0 until targets.length) targets(u) = Tensor[T](compTargetSize)

    var curCp = 1
    var ith = 1
    while(ith <= allInBatch){
      if(ith % comp == 1){
        var b = 0
        while(b < inputData.size) {
          inputs(b).update(curCp,inputData(b).narrow(1, ith, 1))
          b += 1
        }
        b = 0
        while(b < targetData.size) {
          targets(b).update(curCp,targetData(b).narrow(1, ith, 1))
          b += 1
        }
        curCp += 1
      }
      ith += 1
    }
    MiniBatch(inputs,targets)
  }

  def getOrig(): MiniBatch[T] = {
    val inputs = new Array[Tensor[T]](inputData.length)
    val targets = new Array[Tensor[T]](targetData.length)
    val allInBatch = inputData.head.size(1)

    val origFeatureSize = inputData.head.size()
    origFeatureSize(0) = allInBatch - allInBatch / comp
    val origTargetSize = targetData.head.size()
    origTargetSize(0) = allInBatch - allInBatch / comp
    for(u <- 0 until inputs.length) inputs(u) = Tensor[T](origFeatureSize)
    for(u <- 0 until targets.length) targets(u) = Tensor[T](origTargetSize)
    var curCp = 1
    var ith = 1
    while(ith <= allInBatch){
      if(ith % comp != 1){
        var b = 0
        while(b < inputData.size) {
          inputs(b).update(curCp,inputData(b).narrow(1, ith, 1))
          b += 1
        }
        b = 0
        while(b < targetData.size) {
          targets(b).update(curCp,targetData(b).narrow(1, ith, 1))
          b += 1
        }
        curCp += 1
      }
      ith += 1
    }
    MiniBatch(inputs,targets)
  }

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    val inputs = new Array[Tensor[T]](inputData.length)
    val targets = new Array[Tensor[T]](targetData.length)
    var b = 0
    while(b < inputData.size) {
      inputs(b) = inputData(b).narrow(1, offset, length)
      b += 1
    }
    b = 0
    while(b < targetData.size) {
      targets(b) = targetData(b).narrow(1, offset, length)
      b += 1
    }

    new CompMiniBatch(inputs, targets,comp = this.comp)
  }

  override def getInput(): Activity = {
    input
  }

  override def getTarget(): Activity = {
    target
  }

  override def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.length > 0, "samples is empty")
    require(comp > 0 && samples.length % comp == 0,"comp -> orig should the same")
    require(batchSize == 0 || samples.length <= batchSize, "setValue: samples's size doesn't " +
      s"match mini batch size, excepted ${size() } got ${samples.length}")
    val resize = batchSize != samples.length  || featurePaddingParam.isDefined ||
      labelPaddingParam.isDefined || size() != samples.length
    if (batchSize == 0) {
      batchSize = samples.length // set a batchSize when set data.
      unlabeled = samples.head.numLabel() == 0
    }

    val longestFeature = if (featurePaddingParam.isDefined) {
      Some(MiniBatch.findLongestFeatures(samples))
    } else {
      None
    }

    val longestLabel = if (featurePaddingParam.isDefined) {
      Some(MiniBatch.findLongestLabels(samples))
    } else {
      None
    }

    if (resize) {
      MiniBatch.resize(samples, this, featurePaddingStrategy,
        labelPaddingStrategy, featurePadding, labelPadding,
        longestFeature, longestLabel)
    }

    MiniBatch.copyWithPadding[T](samples, this, unlabeled,
      featurePadding, labelPadding)
    this
  }

  @deprecated("Old interface", "0.2.0")
  override def data(): Tensor[T] = {
    require(targetData.length == 1, "Deprecated method," +
      " Only support TensorMiniBatch.")
    input.asInstanceOf[Tensor[T]]
  }

  @deprecated("Old interface", "0.2.0")
  override def labels(): Tensor[T] = {
    require(inputData.length == 1, "Deprecated method," +
      " Only support TensorMiniBatch.")
    target.asInstanceOf[Tensor[T]]
  }

}
