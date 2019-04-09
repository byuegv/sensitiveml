package slpart.slalexnet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgToBatch, BGRImgToSample, BytesToBGRImg}
import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD.{EpochSchedule, Regime}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T, Table}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import slpart.SLTransformer.{LabeledPointToSample, SampleToLabeledPoint}
import slpart.datatreating._

object TrainAlexnetCifar10 {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)
  import slpart.utils.Utils._

  def main(args: Array[String]): Unit ={
    trainParser.parse(args,new TrainParams()).map(param =>{
      if(param.summaryPath.isDefined) LoggerFilter.redirectSparkInfoLogs(param.summaryPath.get+"/bigdl.log")
      else LoggerFilter.redirectSparkInfoLogs()

      val conf = Engine.createSparkConf()
        .setAppName(param.appname)
      val sc = new SparkContext(conf)
      Engine.init

      val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
      val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
      val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
      val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

      val dataDir = param.folder
      val testDataPath = dataDir + "/test_batch.bin"
      val trainDataPath = Array(
        dataDir + "/data_batch_1.bin",
        dataDir + "/data_batch_2.bin",
        dataDir + "/data_batch_3.bin",
        dataDir + "/data_batch_4.bin",
        dataDir + "/data_batch_5.bin"
      )

      val trainFeaLab = if(param.featureNormalize.trim.toBoolean){
        System.out.println("z-score features...")
        SampleToLabeledPoint().apply(BGRImgToSample().apply(BGRImgNormalizer(trainMean,trainStd).apply(
          BytesToBGRImg().apply(Cifar10ByteRecord.loadTrain(trainDataPath: _*).iterator)
        ))).toArray
      }
      else{
        Cifar10.loadTrainData(trainDataPath: _*)
      }

      val testFeaLab = if(param.featureNormalize.trim.toBoolean){
        SampleToLabeledPoint().apply(BGRImgToSample().apply(BGRImgNormalizer(testMean,testStd).apply(
          BytesToBGRImg().apply(Cifar10ByteRecord.loadTest(testDataPath).iterator)
        ))).toArray
      }
      else{
        Cifar10.loadTestData(testDataPath)
      }

//      val trainSamples = DataSet.array(Cifar10ByteRecord.loadTrain(trainDataPath: _*),sc) ->
//        BytesToBGRImg() -> BGRImgNormalizer(trainMean,trainStd) -> BGRImgToSample() ->
//        SampleToLabeledPoint() -> LabeledPointToSample() -> SampleToMiniBatch(param.batchSize)
//
//      val testSamples = DataSet.array(Cifar10ByteRecord.loadTest(testDataPath),sc) ->
//        BytesToBGRImg() -> BGRImgNormalizer(testMean,testStd) -> BGRImgToSample() ->
//        SampleToLabeledPoint() -> LabeledPointToSample() -> SampleToMiniBatch(param.batchSize)


//      val x = LabeledPointToSample().apply(SampleToLabeledPoint().apply(
//        BGRImgToSample().apply(BGRImgNormalizer(testMean,testStd).apply(BytesToBGRImg().apply(
//          sc.parallelize(Cifar10ByteRecord.loadTest(testDataPath))
//        )))
//      ))

//      // 获取LabeledPoint形式的数据
//      val trainFeaLab = Cifar10.loadTrainData(trainDataPath: _*)
//      val testFeaLab = Cifar10.loadTestData(testDataPath)
//

      val trainSamples = if(param.usecomp.trim.toBoolean){
        // 获取压缩之后的训练数据
        val startComp = System.nanoTime()
        val trainSamplesT = CompDeal.getCompTrainData(sc.parallelize(trainFeaLab.sortWith(_.label < _.label)),
          param.itqbitN,param.itqitN,param.itqratioN,param.upBound,param.splitN,param.isSparse)
        System.out.println(s"comp cost Time:${(System.nanoTime() - startComp)*1.0/1e9} seconds")
        trainSamplesT
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
        AlexNetForCIFAR10(classNum = param.classNumber)
      }

      // load or create optimMethod
      val optimMethod = if(param.loadSnapshot.trim.toBoolean && param.stateSnapshot.isDefined){
        System.out.println(s"load stateSnapshot from ${param.stateSnapshot.get}")
        OptimMethod.load[Float](param.stateSnapshot.get)
      }
      else{
        val optmethod = param.optmethod.trim().toLowerCase
        val omethod = optmethod match {
          case "sgd" => new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
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

      val prestate = T(("taskdelratio",param.taskdelratio),
        ("taskdropdecay",param.taskdropdecay),
        ("epochdelratio",param.epochdelratio),
        ("epochdropdecay",param.epochdropdecay),
        ("nodelepoch",param.nodelepoch),
        ("trainmode",param.trainmode.trim().toLowerCase()),
        ("batchsize",param.batchSize),
        ("taskdel",param.taskdel.trim().toBoolean),
        ("epochdel",param.epochdel),
        ("layername","conv_1"),
        ("gradname","gradWeight"),
        ("usecomp",param.usecomp.trim.toBoolean),
        ("inner",param.inner.trim.toBoolean),
        ("copymodel",param.copymodel.trim().toBoolean),
        ("taskdelstrategy",param.taskdelstrategy))
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
        val trainSummary = TrainSummary(param.summaryPath.get,param.appname)
        val validationSummary = ValidationSummary(param.summaryPath.get,param.appname)
        trainSummary.setSummaryTrigger("LearningRate",Trigger.everyEpoch)
        trainSummary.setSummaryTrigger("Parameters",Trigger.everyEpoch)
        trainSummary.setSummaryTrigger("Iterationdelratio",Trigger.severalIteration(1))
        trainSummary.setSummaryTrigger("Timeratio",Trigger.severalIteration(1))
        trainSummary.setSummaryTrigger("CompMean",Trigger.severalIteration(1))
        trainSummary.setSummaryTrigger("TrainMean",Trigger.severalIteration(1))
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
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))

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
