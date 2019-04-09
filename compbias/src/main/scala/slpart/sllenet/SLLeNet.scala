package slpart.sllenet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.numeric.NumericFloat

object SLLeNet {
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv_1")) //(28-5+2*0)/1 +1 = 24    24x24x6
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_1"))//(24-2+2*0)/2 + 1 = 12     12x12x6
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv_2"))//(12-5+2*0)/1  +1 = 8    8x8x12
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2).setName("pool_2")) //(8-2+2*0)/2 +1 = 4     4x4x12
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc_3"))
      .add(Tanh())
      .add(Linear(100, classNum).setName("fc_4"))
      .add(LogSoftMax())
    model
  }
  def modelInit(model: Module[Float]): Unit = {
    def initModules(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float]
        => container.modules.foreach(m => initModules(m))
        case spatialConvolution
          if (spatialConvolution.isInstanceOf[SpatialConvolution[Float]]) =>
          val curModel = spatialConvolution.asInstanceOf[SpatialConvolution[Float]]
          val n: Float = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toFloat)
          curModel.bias.apply1(_ => 0.0f)
        case spatialBatchNormalization
          if (spatialBatchNormalization.isInstanceOf[SpatialBatchNormalization[Float]]) =>
          val curModel = spatialBatchNormalization.asInstanceOf[SpatialBatchNormalization[Float]]
          curModel.weight.apply1(_ => 1.0f)
          curModel.bias.apply1(_ => 0.0f)
        case linear if (linear.isInstanceOf[Linear[Float]]) =>
          linear.asInstanceOf[Linear[Float]].bias.apply1(_ => 0.0f)
        case _ => Unit
      }
    }
    initModules(model)
  }
}
