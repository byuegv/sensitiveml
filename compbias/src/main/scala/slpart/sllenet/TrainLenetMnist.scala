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
import org.apache.spark.mllib.regression.LabeledPoint
import slpart.SLTransformer.{LabeledPointToSample, SampleToLabeledPoint}
import slpart.datatreating._

import scala.collection.mutable.ArrayBuffer

object TrainLenetMnist {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)

  import Utils._

  def main(args: Array[String]): Unit ={
    trainParser.parse(args,new TrainParams()).map(param =>{
      if(param.summaryPath.isDefined) LoggerFilter.redirectSparkInfoLogs(param.summaryPath.get+"/bigdl.log")
      else LoggerFilter.redirectSparkInfoLogs()
      val conf = Engine.createSparkConf()
        .setAppName(param.appName)
      val sc = new SparkContext(conf)
      Engine.init

      val dataDir = param.folder
      val testImagePath = dataDir + "/t10k-images-idx3-ubyte"
      val testLabelPath = dataDir + "/t10k-labels-idx1-ubyte"
      val trainImagePath = dataDir + "/train-images-idx3-ubyte"
      val trainLabelPath = dataDir + "/train-labels-idx1-ubyte"

      val trainFeaLab = if(param.zScore.trim.toBoolean){
        System.out.println("z-score features...")
        SampleToLabeledPoint().apply(GreyImgToSample().apply(GreyImgNormalizer(trainMean,trainStd).apply(
          BytesToGreyImg(28,28).apply(MnistByteRecord.load(trainImagePath,trainLabelPath).iterator)
        ))).toArray
      }
      else{
        MnistData.getTrainLabeledPoint(trainImagePath,trainLabelPath)
      }
      val splitTrainFeaLab = new Array[ArrayBuffer[LabeledPoint]](10)
      for(u <- 0 until 10) splitTrainFeaLab(u) = ArrayBuffer[LabeledPoint]()
      for(ix <- 0 until trainFeaLab.length){
        val lf = trainFeaLab(ix)
        val cat = (lf.label + 0.5).toInt
        splitTrainFeaLab(cat - 1) += lf
      }

      val testFeaLab = if(param.zScore.trim.toBoolean){
        SampleToLabeledPoint().apply(GreyImgToSample().apply(GreyImgNormalizer(testMean,testMean).apply(
          BytesToGreyImg(28,28).apply(MnistByteRecord.load(testImagePath,testLabelPath).iterator)
        ))).toArray
      }
      else{
        MnistData.getTestLabeledPoint(testImagePath,testLabelPath)
      }

      val trainSamples = if(param.useComp.trim.toBoolean){
        // 获取压缩之后的训练数据
        val startComp = System.nanoTime()
        val res = (1 to 10).map(ix => CompDeal.getCompTrainDataZip(sc.parallelize(splitTrainFeaLab(ix - 1)),
          param.itqbitN,param.itqitN,param.itqratioN,param.upBound,param.splitN,param.isSparse)).reduce((lf,rg) => lf.union(rg))
        System.out.println(s"comp cost Time:${(System.nanoTime() - startComp)*1.0/1e9} seconds")
        res
      }
      else{
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

      // load or create model
      val model = if(param.loadSnapshot.trim.toBoolean && param.modelSnapshot.isDefined){
        System.out.println(s"load model from ${param.modelSnapshot.get}")
        Module.load[Float](param.modelSnapshot.get)
      }
      else{
        SLLeNet(classNum = param.classNumber)
      }

      // load or create optimMethod
      val optimMethod = if(param.loadSnapshot.trim.toBoolean && param.stateSnapshot.isDefined){
        System.out.println(s"load stateSnapshot from ${param.stateSnapshot.get}")
        OptimMethod.load[Float](param.stateSnapshot.get)
      }
      else{
        val optmethod = param.optMethod.trim().toLowerCase
        val omethod = optmethod match {
          case "sgd" =>  new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,momentum = param.momentum,nesterov = param.nesterov)
          case "adam" => new Adam[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case "adadelta" => new Adadelta[Float]()
          case "rmsprop" => new RMSprop[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case "adamax" => new Adamax[Float](learningRate = param.learningRate)
          case "adagrad" => new Adagrad[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case "lbfgs" => new LBFGS[Float](learningRate = param.learningRate)
          case "ftrl" => new Ftrl[Float](learningRate = param.learningRate)
          case _ => new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,momentum = param.momentum,nesterov = param.nesterov)
        }
        omethod
      }

      //save initial model and optimmethod
      if(param.storeInitModel.trim.toBoolean){
        if(param.storeInitModelPath.isDefined){
          System.out.println(s"save initial model in ${param.storeInitModelPath.get}")
          model.save(param.storeInitModelPath.get,true)
        }
        if(param.storeInitStatePath.isDefined){
          System.out.println(s"save initial state in ${param.storeInitStatePath.get}")
          optimMethod.save(param.storeInitStatePath.get,true)
        }
      }

      // create optimizer
      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainSamples,
        criterion = new ClassNLLCriterion[Float](),
        batchSize = param.batchSize
      )

      val prestate = T(("taskDel",param.taskDel.trim.toBoolean),
        ("taskStrategy",param.taskStrategy.trim.toLowerCase()),
        ("taskRatio",param.taskRatio),
        ("trakDrop",param.taskDrop),

        ("epochDel",param.epochDel.trim.toBoolean),
        ("epochStrategy",param.epochStrategy.trim().toLowerCase()),
        ("epochRatio",param.epochRatio),
        ("epochDrop",param.epochDrop),

        ("notIteration",param.notIteration),
        ("layerName","conv_1"),
        ("gradName","gradWeight"),

        ("useComp",param.useComp.trim.toBoolean),
        ("getGradient",param.getGradient.trim.toBoolean)
       )
      // set user defined state
      optimizer.setState(prestate)

      // save checkpoint
      if(param.checkpoint.isDefined){
        optimizer.setCheckpoint(param.checkpoint.get,Trigger.severalIteration(param.checkpointIteration))
      }
      if(param.overWriteCheckpoint){
        optimizer.overWriteCheckpoint()
      }

      // get train or validation summary
      if(param.summaryPath.isDefined){
        val trainSummary = TrainSummary(param.summaryPath.get,param.appName)
        val validationSummary = ValidationSummary(param.summaryPath.get,param.appName)
        trainSummary.setSummaryTrigger("LearningRate",Trigger.everyEpoch)
//        trainSummary.setSummaryTrigger("Parameters",Trigger.everyEpoch)
//        trainSummary.setSummaryTrigger("Iterationdelratio",Trigger.severalIteration(1))
//        trainSummary.setSummaryTrigger("Timeratio",Trigger.severalIteration(1))
//        trainSummary.setSummaryTrigger("CompMean",Trigger.severalIteration(1))
//        trainSummary.setSummaryTrigger("TrainMean",Trigger.severalIteration(1))
        optimizer.setTrainSummary(trainSummary)
        optimizer.setValidationSummary(validationSummary)
        model.toGraph().saveGraphTopology(param.summaryPath.get)
      }
      optimizer.setValidation(
        trigger = Trigger.everyEpoch,
        sampleRDD = testSamples,
        vMethods = Array(new Top1Accuracy, new Top5Accuracy, new Loss),
        batchSize = param.batchSize
      )
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxIteration(param.maxIteration))

      val trainedModel = optimizer.optimize()

      // save trained model and states
      if(param.storeTrainedModel.trim.toBoolean){
        if(param.storeTrainedModelPath.isDefined) {
          System.out.println(s"save trained model in ${param.storeTrainedModelPath.get}")
          trainedModel.save(param.storeTrainedModelPath.get,overWrite = true)
        }
        if(param.storeTrainedStatePath.isDefined) {
          System.out.println(s"save trained state in ${param.storeTrainedStatePath.get}")
          optimMethod.save(param.storeTrainedStatePath.get,overWrite = true)
        }
      }

      sc.stop()

    })
  }
}
