package slpart.sllenet

import scopt.OptionParser

object Utils {
  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  case class TrainParams(
                          appName: String = "appname",
                          folder: String = "./",
                          checkpoint: Option[String] = None,
                          loadSnapshot: String = "false",
                          modelSnapshot: Option[String] = None,
                          stateSnapshot: Option[String] = None,
                          summaryPath: Option[String] = None,
                          batchSize: Int = 128,
                          maxEpoch: Int = 100,
                          overWriteCheckpoint: Boolean = false,
                          learningRate: Double = 0.001,
                          learningRateDecay: Double = 0.0,
                          weightDecay: Double = 0.0,
                          graphModel: Boolean = false,
                          maxIteration: Int = 100000,
                          momentum: Double = 0.0,
                          dampening: Double = 0.0,
                          nesterov: Boolean = false,
                          classNumber: Int = 10,
                          checkpointIteration: Int = 10000,
                          maxLr: Double = 0.06,
                          warmupEpoch: Option[Int] = None,
                          gradientL2NormThreshold: Option[Double] = None,
                          itqbitN: Int = 1,
                          itqitN: Int = 20,
                          itqratioN: Int = 100,
                          minPartN: Int = 1,
                          upBound: Int = 30,
                          splitN : Double = 2.0,
                          isSparse: Boolean = false,
                          taskDel: String = "true",
                          taskStrategy: String = "default",
                          taskRatio: Double = 0.01,
                          taskDrop: Double = 0.05,
                          epochDel: String = "false",
                          epochStrategy: String = "default",
                          epochRatio: Double = 0.001,
                          epochDrop: Double = 0.05,
                          notIteration: Int = 0,
                          optMethod: String = "sgd",
                          useComp: String = "false",
                          getGradient: String = "false",
                          storeInitModel: String = "false",
                          storeInitModelPath: Option[String] = None,
                          storeInitStatePath: Option[String] = None,
                          storeTrainedModel: String = "false",
                          storeTrainedModelPath: Option[String] = None,
                          storeTrainedStatePath: Option[String] = None,
                          zScore: String = "false",
                          lrScheduler: Option[String] = None
                        )

  val trainParser = new OptionParser[TrainParams]("Training Example") {
    opt[String]("appName")
      .text("the application name")
      .action((x,c) => c.copy(appName = x))
    opt[String]('f', "folder")
      .text("where you put the  data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("loadSnapshot")
      .text("load mode state from snapshot")
      .action((x,c) => c.copy(loadSnapshot = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model and state")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[String]("summary")
      .text("where to store the training summary")
      .action((x, c) => c.copy(summaryPath = Some(x)))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action( (_, c) => c.copy(overWriteCheckpoint = true) )
    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]('l', "learningRate")
      .text("inital learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("learning rate decay")
      .action((x,c) => c.copy(learningRateDecay = x))
    opt[Unit]('g', "graphModel")
      .text("use graph model")
      .action((x, c) => c.copy(graphModel = true))
    opt[Unit]("nesterov")
      .text("nesterov default = false")
      .action((_,c) => c.copy(nesterov = true))
    opt[Int]('i', "maxIteration")
      .text("iteration numbers")
      .action((x, c) => c.copy(maxIteration = x))
    opt[Int]("classNum")
      .text("class number")
      .action((x, c) => c.copy(classNumber = x))
    opt[Int]("checkpointIteration")
      .text("checkpoint interval of iterations")
      .action((x, c) => c.copy(checkpointIteration = x))
    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]("maxLr")
      .text("max Lr after warm up")
      .action((x, c) => c.copy(maxLr = x))
    opt[Int]("warmupEpoch")
      .text("warm up epoch numbers")
      .action((x, c) => c.copy(warmupEpoch = Some(x)))
    opt[Double]("gradientL2NormThreshold")
      .text("gradient L2-Norm threshold")
      .action((x, c) => c.copy(gradientL2NormThreshold = Some(x)))

    opt[Int]("itqbitN")
      .text("itqbitN")
      .action((x,c) => c.copy(itqbitN = x))
    opt[Int]("itqitN")
      .text("itqitN")
      .action((x,c) => c.copy(itqitN = x))
    opt[Int]("itqrationN")
      .text("itqratioN")
      .action((x,c) => c.copy(itqratioN = x))
    opt[Int]("minPartN")
      .text("minPartN")
      .action((x,c) => c.copy(minPartN = x))
    opt[Int]("upBound")
      .text("upBound")
      .action((x,c) => c.copy(upBound = x))
    opt[Double]("splitN")
      .text("splitN")
      .action((x,c) => c.copy(splitN = x))
    opt[Unit]("isSparse")
      .text("isSparse")
      .action((_, c) => c.copy(isSparse = true))

    opt[String]("taskDel")
      .text("task level filter")
      .action((x,c) => c.copy(taskDel = x))
    opt[String]("taskStrategy")
      .text("task level del strategy")
      .action((x,c) => c.copy(taskStrategy = x))
    opt[Double]("taskRatio")
      .text("task delete ratio")
      .action((x,c) => c.copy(taskRatio = x))
    opt[Double]("taskDrop")
      .text("task delete drop decay")
      .action((x,c) => c.copy(taskDrop = x))

    opt[String]("epochDel")
      .text("epoch delete")
      .action((x,c) => c.copy(epochDel = x))
    opt[String]("epochStrategy")
      .text("epoch level del strategy")
      .action((x,c) => c.copy(epochStrategy = x))
    opt[Double]("epochRatio")
      .text("epoch delete ratio")
      .action((x,c) => c.copy(epochRatio = x))
    opt[Double]("epochDrop")
      .text("epoch delete drop decay")
      .action((x,c) => c.copy(epochDrop= x))

    opt[Int]("notIteration")
      .text("do not del anythig in fist  ith epoch")
      .action((x,c) => c.copy(notIteration = x))
    opt[String]( "optMethod")
      .text("optimization metho default is sgd")
      .action((x, c) => c.copy(optMethod = x))
    opt[String]("useComp")
      .text("if use compressed data default if false")
      .action((x,c) => c.copy(useComp = x))
    opt[String]("getGradient")
      .text("get gradients")
      .action((x,c) => c.copy(getGradient = x))

    opt[String]("storeInitModel")
      .text("if save store init model")
      .action((x,c) => c.copy(storeInitModel = x))
    opt[String]("storeInitModelPath")
      .text("save init model in ...")
      .action((x,c) => c.copy(storeInitModelPath = Some(x)))
    opt[String]("storeInitStatePath")
      .text("save init state in ...")
      .action((x,c) => c.copy(storeInitStatePath = Some(x)))
    opt[String]("storeTrainedModel")
      .text("if save trained model")
      .action((x,c) => c.copy(storeTrainedModel = x))
    opt[String]("storeTrainedModelPath")
      .text("save trained model in ...")
      .action((x,c) => c.copy(storeTrainedModelPath = Some(x)))
    opt[String]("storeTrainedStatePath")
      .text("storeTrainedStatePath")
      .action((x,c) => c.copy(storeTrainedStatePath = Some(x)))

    opt[String]("zScore")
      .text("if z-score features")
      .action((x,c) => c.copy(zScore = x))
    opt[String]("lrScheduler")
      .text("sgd learningRate scheduler")
      .action((x,c) => c.copy(lrScheduler = Some(x)))
  }

  case class TestParams(
                         folder: String = "./",
                         model: String = "",
                         batchSize: Int = 128
                       )

  val testParser = new OptionParser[TestParams]("Test Example") {
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

}
