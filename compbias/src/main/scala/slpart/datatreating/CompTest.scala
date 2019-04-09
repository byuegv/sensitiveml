package slpart.datatreating

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToSample}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level
import com.intel.analytics.bigdl.numeric.NumericFloat
import org.apache.spark.mllib.regression.LabeledPoint

import slpart.SLTransformer.SampleToLabeledPoint


object CompTest {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("compTest")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)
    Engine.init

    val trainMean = 0.13066047740239506
    val trainStd = 0.3081078

    val testMean = 0.13251460696903547
    val testStd = 0.31048024


    val dataDir = "/home/hadoop/Documents/accML-Res-Store/mnist"
    val testImagePath = dataDir + "/t10k-images-idx3-ubyte"
    val testLabelPath = dataDir + "/t10k-labels-idx1-ubyte"
    val trainImagePath = dataDir + "/train-images-idx3-ubyte"
    val trainLabelPath = dataDir + "/train-labels-idx1-ubyte"



    val transformer =  BytesToGreyImg(28,28) -> GreyImgNormalizer(trainMean,trainStd) -> GreyImgToSample() ->
      SampleToLabeledPoint()

    val trainData =transformer.apply(MnistByteRecord.load(trainImagePath,trainLabelPath).toIterator).toArray

    //System.out.println(s"feature:\n${trainData.head.features}\nlabel:\n${trainData.head.label}")
    //System.exit(0)

    val stTime = System.nanoTime()
    val compTrain = CompDeal.getCompTrainDataPartition(sc.parallelize(trainData.sortWith((x,y) => x.label < y.label)))

    val count = compTrain.count()
    println(s"count:${count}")

//    compTrain.take(1).foreach(x => {
//      System.out.println(s"features:\n${x.feature()}\nlabel:${x.label()}")
//    })
    System.out.println(s"cost: ${(System.nanoTime()-stTime)*1.0f/1e9} s")
    sc.stop()
  }

}
