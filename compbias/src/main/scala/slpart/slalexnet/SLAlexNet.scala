package slpart.slalexnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat


object AlexNetForImageNet{
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,224,224)))
    model.add(SpatialConvolution(3,96,11,11,4,4,3,3).setName("conv_1")) // (224-11+2*3)/4 + 1 = 55 55x55x96
      .add(SpatialBatchNormalization(96))
      .add(ReLU(true).setName("relu_1"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_1"))//(55-3+2*0)/2 + 1 = 27 27x27x96
      .add(SpatialConvolution(96,256,5,5,1,1,2,2).setName("conv_2")) // (27-5+2*2)/1 + 1 = 27 27x27x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_2"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_2"))//(27-3+2*0)/2 + 1 = 27 13x13x256
      .add(SpatialConvolution(256,384,3,3,1,1,1,1).setName("conv_3")) // (13-3+2*1)/1 + 1 = 13 13x13x384
      .add(SpatialBatchNormalization(384))
      .add(ReLU(true).setName("relu_3"))
      .add(SpatialConvolution(384,384,3,3,1,1,1,1).setName("conv_4")) // (13-3+2*1)/1 + 1 = 13 13x13x384
      .add(SpatialBatchNormalization(384))
      .add(ReLU(true).setName("relu_4"))
      .add(SpatialConvolution(384,256,3,3,1,1,1,1).setName("conv_5")) // (13-3+2*1)/1 + 1 = 13 13x13x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_5"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_5")) // (13-3+2*0)/2 + 1 = 13 6x6x256
      .add(Reshape(Array(6*6*256)))
      .add(Linear(6*6*256,4096).setName("fc_6"))
      .add(ReLU(true).setName("relu_6"))
      .add(Dropout(0.5).setName("dropout_6"))
      .add(Linear(4096,4096).setName("fc_7"))
      .add(ReLU(true).setName("relu_7"))
      .add(Dropout(0.5).setName("dropout_7"))
      .add(Linear(4096,classNum).setName("fc_8"))
      .add(LogSoftMax())
    model
  }
}

object AlexNetForMNIST2{
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1,28,28)))
    model.add(SpatialConvolution(1,64,3,3,1,1).setName("conv_1")) // (28-3+2*0)/1 + 1 = 26 26x26x64
      .add(SpatialBatchNormalization(64))
      .add(ReLU(true).setName("relu_1"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_1"))//(26-3+2*0)/2 + 1 =  12x12x64
      .add(SpatialConvolution(64,128,3,3,1,1,1,1).setName("conv_2")) // (12-3+2*1)/1 + 1 = 12 12x12x128
      .add(SpatialBatchNormalization(128))
      .add(ReLU(true).setName("relu_2"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_2"))//(12-3+2*0)/2 + 1 = 5 5x5x128
      .add(SpatialConvolution(128,256,3,3,1,1,1,1).setName("conv_3")) // (5-3+2*1)/1 + 1 = 5 5x5x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_3"))
      .add(SpatialConvolution(256,256,3,3,1,1,1,1).setName("conv_4")) // (5-3+2*1)/1 + 1 = 5 5x5x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_4"))
      .add(SpatialConvolution(256,128,3,3,1,1,1,1).setName("conv_5")) // (5-3+2*1)/1 + 1 = 5 5x5x128
      .add(SpatialBatchNormalization(128))
      .add(ReLU(true).setName("relu_5"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_5")) // (5-3+2*0)/2 + 1 = 2 2x2x128
      .add(Reshape(Array(2*2*128)))
      .add(Linear(2*2*128,128).setName("fc_6"))
      .add(ReLU(true).setName("relu_6"))
      .add(Dropout(0.5).setName("dropout_6"))
      .add(Linear(128,128).setName("fc_7"))
      .add(ReLU(true).setName("relu_7"))
      .add(Dropout(0.5).setName("dropout_7"))
      .add(Linear(128,classNum).setName("fc_8"))
      .add(LogSoftMax())
    model
  }
}

object AlexNetForMNIST {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1,64,3,3,1,1).setName("conv_1"))// (28-3)/1 + 1 = 26 26x26x64
      .add(SpatialBatchNormalization(64))
      .add(ReLU(true).setName("relu_1"))
      .add(SpatialMaxPooling(3,3,2,2).setName("maxpool_1"))//(26-3)/2 + 1 = 12 12x12x64
      .add(SpatialConvolution(64,128,3,3,1,1).setName("conv_2"))//(12-3)/1 + 1 = 10 10x10x128
      .add(SpatialBatchNormalization(128))
      .add(ReLU(true).setName("relu_2"))
      .add(SpatialMaxPooling(3,3,2,2).setName("maxpool_2"))//(10-3)/2 + 1 = 4 4x4x128
      .add(SpatialConvolution(128,256,3,3,1,1,1,1).setName("conv_3"))//(4-3 + 2*1)/1 + 1 = 2 4x4x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_3"))
      .add(SpatialMaxPooling(2,2,1,1).setName("maxpool_3"))//(4-2)/1 + 1 = 3 3x3x256
      .add(Reshape(Array(3*3*256)))
      .add(Linear(3*3*256,1024).setName("fc_4"))
      .add(ReLU(true).setName("relu_4"))
      .add(Dropout(0.5).setName("dropout_4"))
      .add(Linear(1024,1024).setName("fc_5"))
      .add(ReLU(true).setName("relu_5"))
      .add(Dropout(0.5).setName("dropout_5"))
      .add(Linear(1024,classNum).setName("fc_6"))
      .add(LogSoftMax())

    model
  }
}

object AlexNetForCIFAR102{
  def apply(classNum: Int = 10): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,64,3,3,1,1).setName("conv_1")) // (32-3+2*0)/1 + 1 = 32 30x30x64
      .add(SpatialBatchNormalization(64))
      .add(ReLU(true).setName("relu_1"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_1"))//(30-3+2*0)/2 + 1 =  14x14x64
      .add(SpatialConvolution(64,128,3,3,1,1,1,1).setName("conv_2")) // (14-3+2*1)/1 + 1 = 14 14x14x128
      .add(SpatialBatchNormalization(128))
      .add(ReLU(true).setName("relu_2"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_2"))//(14-3+2*0)/2 + 1 = 6 6x6x128
      .add(SpatialConvolution(128,256,3,3,1,1,1,1).setName("conv_3")) // (6-3+2*1)/1 + 1 = 6 6x6x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_3"))
      .add(SpatialConvolution(256,256,3,3,1,1,1,1).setName("conv_4")) // (6-3+2*1)/1 + 1 = 6 6x6x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_4"))
      .add(SpatialConvolution(256,128,3,3,1,1,1,1).setName("conv_5")) // (6-3+2*1)/1 + 1 = 6 6x6x128
      .add(SpatialBatchNormalization(128))
      .add(ReLU(true).setName("relu_5"))
      .add(SpatialMaxPooling(3,3,2,2).setName("pool_5")) // (6-3+2*0)/2 + 1 = 2 2x2x128
      .add(Reshape(Array(2*2*128)))
      .add(Linear(2*2*128,128).setName("fc_6"))
      .add(ReLU(true).setName("relu_6"))
      .add(Dropout(0.5).setName("dropout_6"))
      .add(Linear(128,128).setName("fc_7"))
      .add(ReLU(true).setName("relu_7"))
      .add(Dropout(0.5).setName("dropout_7"))
      .add(Linear(128,classNum).setName("fc_8"))
      .add(LogSoftMax())
    model
  }
}



object AlexNetForCIFAR10 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(Reshape(Array(3,32,32)))
    model.add(SpatialConvolution(3,64,3,3,1,1).setName("conv_1"))// (32-3)/1 + 1 = 30 30x30x64
      .add(SpatialBatchNormalization(64))
      .add(ReLU(true).setName("relu_1"))
      //      .add(SpatialCrossMapLRN(5,0.0001,0.75).setName("norm_1"))
      .add(SpatialMaxPooling(3,3,2,2).setName("maxpool_1"))//(30-3)/2 + 1 = 14 14x14x64
      .add(SpatialConvolution(64,128,3,3,1,1).setName("conv_2"))//(14-3)/1 + 1 = 12 12x12x128
      .add(SpatialBatchNormalization(128))
      .add(ReLU(true).setName("relu_2"))
      //      .add(SpatialCrossMapLRN(5,0.0001,0.75).setName("norm_2"))
      .add(SpatialMaxPooling(3,3,2,2).setName("maxpool_2"))//(12-3)/2 + 1 = 5 5x5x128
      .add(SpatialConvolution(128,256,3,3,1,1,1,1).setName("conv_3"))//(5-3 + 2*1)/1 + 1 = 5 5x5x256
      .add(SpatialBatchNormalization(256))
      .add(ReLU(true).setName("relu_3"))
      //      .add(SpatialCrossMapLRN(5,0.0001,0.75).setName("norm_3"))
      .add(SpatialMaxPooling(2,2,1,1).setName("maxpool_3"))//(5-2)/1 + 1 = 4 4x4x256
      .add(Reshape(Array(4096)))
      .add(Linear(4096,1024).setName("fc_4"))
      .add(ReLU(true).setName("relu_4"))
      .add(Dropout(0.5).setName("dropout_4"))
      .add(Linear(1024,1024).setName("fc_5"))
      .add(ReLU(true).setName("relu_5"))
      .add(Dropout(0.5).setName("dropout_5"))
      .add(Linear(1024,classNum).setName("fc_6"))
      .add(LogSoftMax())
    model
  }
}
