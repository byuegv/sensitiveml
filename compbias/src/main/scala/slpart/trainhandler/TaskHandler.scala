package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.{Module, _}
import com.intel.analytics.bigdl.dataset.MiniBatch
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, _}
import scala.collection.mutable.{ArrayBuffer, Map}

class TaskHandler[T](subModelId: Int,
                     localModel: Module[T],
                     localCriterion: Criterion[T],
                     miniBatch: MiniBatch[T],
                     layerName: String,
                     gradName: String) {
  val input = miniBatch.getInput().toTensor
  val target = miniBatch.getTarget().toTensor
  val batchSize = input.size(1)
  val compOrigSize = target.size(2)
  val featureSize = input.size(2) / target.size(2)

  //用于存储梯度信息(compId,index,grad)
  val gradInfoAry = new ArrayBuffer[(Int,Int,Double)]

  /**
    * 训练压缩点并保存梯度信息
    */
  def trainCompPoints() = {
    val compInput = input.narrow(2,1,featureSize)
    val compTarget = target.narrow(2,1,1)
    for(cur <- 1 to batchSize){
      val oneinput = compInput.narrow(1,cur,1)
      val idLabel = compTarget.narrow(1,cur,1).squeeze().toArray()(0)
      val (compId,label) = CompUtils.paraseIdLabel(idLabel)
      val onetarget = Tensor(T(label))


      //确认-上一次计算得到的梯度信息会被保留下来
      val preGradWeight = localModel.getParametersTable()[Table](layerName)[Tensor[T]](gradName).clone()
//      println(s"preGradWeight.abs.sum:${preGradWeight.abs().sum()}")

      val oneoutput = localModel.forward(oneinput)
      localCriterion.forward(oneoutput,onetarget)
      val oneerrors = localCriterion.backward(oneoutput,onetarget)
      localModel.backward(oneinput,oneerrors)

      val trainGradWeight = localModel.getParametersTable()[Table](layerName)[Tensor[T]](gradName).clone()
      val gradWeight = trainGradWeight - preGradWeight
      var grad = gradWeight.abs().sum().asInstanceOf[Float]
      if(grad.isNaN || grad.isInfinite){
        grad = 1e-20f
      }
      gradInfoAry += Tuple3(compId,cur,grad)
    }
  }

  /**
    * 根据选中的压缩点的(compId,index)选择对应的原始点
    * @param selectAry
    * @return
    */
  def getSelectOrigSamples(selectAry: Array[(Int,Int)]) = {
    val n = selectAry.length
    val origInput = Tensor(n,featureSize * compOrigSize)
    val origTarget = Tensor(n,1 * compOrigSize)
    for(cur <- 1 to n){
      origInput.update(cur,input.narrow(1,selectAry(cur - 1)._2,1))
      origTarget.update(cur,target.narrow(1,selectAry(cur - 1)._2,1))
    }
//    println(s"origInput.szie:${origInput.size().mkString("x")} origTarget.size:${origTarget.size().mkString("x")}" +
//      s" compOrigSize:${compOrigSize} featureSize:${featureSize}")
    val oneOrigInput = origInput.narrow(2,featureSize + 1,(compOrigSize -1 ) * featureSize)
      .reshape(Array(n * (compOrigSize -1),featureSize))
    val oneOrigTarget = origTarget.narrow(2,2,compOrigSize - 1)
      .reshape(Array(n * (compOrigSize -1),1))
    (oneOrigInput,oneOrigTarget)
  }

  /**
    * 返回所有的原始点
    * 并调整格式
    * @return
    */
  def getAllOrigSamples() = {
    val origInput = input.narrow(2,featureSize+1,(compOrigSize - 1) * featureSize)
    val origTarget = target.narrow(2,2,compOrigSize - 1 )
    val oneOrigInput = origInput.reshape(Array(batchSize*(compOrigSize-1),featureSize))
    val oneOrigTarget = origTarget.reshape(Array(batchSize*(compOrigSize-1),1))
    (oneOrigInput,oneOrigTarget)
  }


}
