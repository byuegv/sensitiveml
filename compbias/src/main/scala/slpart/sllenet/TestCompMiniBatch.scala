package slpart.sllenet
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch, GreyImgToSample}
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD.Default
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dataset.CompMiniBatch
import slpart.datatreating._

object TestCompMiniBatch {
  def main(args: Array[String]) = {
    val logger = Logger.getLogger("org")
    logger.setLevel(Level.OFF)
    val conf = Engine.createSparkConf()
      .setAppName("testCompMiniBatch")
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

    val trainFeaLab = MnistData.getTrainLabeledPoint(trainImagePath,trainLabelPath)
    val testFeaLab = MnistData.getTestLabeledPoint(testImagePath,testLabelPath)

    val trainSamples = {
      sc.parallelize(trainFeaLab).map(p =>{
        val fea = p.features.toArray
        val lab = p.label
        Sample(Tensor(T(fea.head,fea.tail: _*)),Tensor(T(lab)))
      })
    }

    // 将验证数据转换为RDD[Sample]
    val testSamples = sc.parallelize(testFeaLab).map(p =>{
      val fea = p.features.toArray
      val lab = p.label
      Sample(Tensor(T(fea.head,fea.tail: _*)),Tensor(T(lab)))
    })

    val model = SLLeNet(10)
    val optiMethod = new SGD()

    // create optimizer
    val optimizer = Optimizer(
      model = model,
      sampleRDD = trainSamples,
      criterion = new ClassNLLCriterion[Float](),
      batchSize = 32 * 4,
      miniBatchImpl = new CompMiniBatch[Float](inputData = Array.tabulate(1)(_ => Tensor(32*4,784)),targetData = Array.tabulate(1)(_ => Tensor(32*4,1)),comp = 4)
    )

    val prestate = T(("taskdelratio",0.0),
      ("taskdropdecay",0.0),
      ("epochdelratio",0.0),
      ("epochdropdecay",0.0),
      ("nodelepoch",0),
      ("trainmode","batch"),
      ("batchsize",32),
      ("taskdel",false),
      ("epochdel","false"),
      ("layername","conv_1"),
      ("gradname","gradWeight"),
      ("usecomp",false),
      ("inner",true),
      ("copymodel",false),
      ("taskdelstrategy","meandiv10"))
    // set user defined state
    optimizer.setState(prestate)

    optimizer.setOptimMethod(optiMethod)
    optimizer.setEndWhen(Trigger.maxEpoch(1))
    optimizer.optimize()
    sc.stop()
  }

}
