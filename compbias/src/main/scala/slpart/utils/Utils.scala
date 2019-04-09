package slpart.utils

import scopt.OptionParser

object Utils {
  case class TrainParams(
                          appname: String = "appname",
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
                          maxIteration: Int = 10000,
                          momentum: Double = 0.0,
                          dampening: Double = 0.0,
                          nesterov: Boolean = false,
                          classNumber: Int = 10,
                          env: String = "local",
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
                          taskdelratio: Double = 0.01,
                          taskdropdecay: Double = 0.05,
                          epochdelratio: Double = 0.001,
                          epochdropdecay: Double = 0.05,
                          nodelepoch: Int = 0,
                          optmethod: String = "sgd",
                          trainmode: String = "batch",
                          markedtrainpath: String = "./mnist-train-data.txt",
                          markedtestpath: String = "./mnist-test-data.txt",
                          idlosspath: String = "./idloss.txt",
                          isfirst: String = "false",
                          sampleratio: Double = 0.1,
                          datasetname: String = "mnist",
                          taskdel: String = "true",
                          epochdel: String = "false",
                          samplemethod: String = "random",
                          usecomp: String = "false",
                          inner: String = "false",
                          copymodel:String = "false",
                          taskdelstrategy:String = "default",
                          storeInitModel: String = "false",
                          storeInitModelPath: Option[String] = None,
                          storeInitStatePath: Option[String] = None,
                          storeTrainedModel: String = "false",
                          storeTrainedModelPath: Option[String] = None,
                          storeTrainedStatePath: Option[String] = None,
                          featureNormalize: String = "false",
                          lrScheduler: Option[String] = None
                        )

  val trainParser = new OptionParser[TrainParams]("Training Example") {
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
      .action( (_, c) => c.copy(isSparse = true) )
    opt[Double]("taskdelratio")
      .text("task delete ratio")
      .action((x,c) => c.copy(taskdelratio = x))
    opt[Double]("taskdropdecay")
      .text("task delete dropdecay")
      .action((x,c) => c.copy(taskdropdecay = x))
    opt[Double]("epochdelratio")
      .text("epoch delete ratio")
      .action((x,c) => c.copy(epochdelratio = x))
    opt[Double]("epochdropdecay")
      .text("epoch delete dropdecay")
      .action((x,c) => c.copy(epochdropdecay = x))
    opt[Int]("nodelepoch")
      .text("do not del anythig in fist  ith epoch")
      .action((x,c) => c.copy(nodelepoch = x))
    opt[String]( "optmethod")
      .text("optimization metho default is sgd")
      .action((x, c) => c.copy(optmethod = x))
    opt[String]("trainmode")
      .text("train model single or batch default is batch")
      .action((x,c) => c.copy(trainmode = x))
    opt[String]("markedtrainpath")
      .text("marked train path")
      .action((x,c) => c.copy(markedtrainpath = x))
    opt[String]("markedtestpath")
      .text("marked test path")
      .action((x,c) => c.copy(markedtestpath = x))
    opt[String]("idlosspath")
        .text("id loss path")
        .action((x,c) => c.copy(idlosspath = x))
    opt[String]("isfirst")
      .text("is first train all data default is false")
      .action((x,c) => c.copy(isfirst = x))
    opt[Double]("sampleratio")
      .text("sampling ratio default is 0.1")
      .action((x,c) => c.copy(sampleratio = x))
    opt[String]("datasetname")
      .text("name of dataset")
      .action((x,c) => c.copy(datasetname = x))
    opt[String]("taskdel")
      .text("task level filter")
      .action((x,c) => c.copy(taskdel = x))
    opt[String]("epochdel")
      .text("epoch level filter")
      .action((x,c) => c.copy(epochdel = x))
    opt[String]("samplemethod")
      .text(("sample method"))
      .action((x,c) => c.copy(samplemethod = x))
    opt[String]("appname")
      .text("the application name")
      .action((x,c) => c.copy(appname = x))
    opt[String]("usecomp")
      .text("if use compressed data default if false")
      .action((x,c) => c.copy(usecomp = x))
    opt[String]("inner")
      .text("dis grad inner")
      .action((x,c) => c.copy(inner = x))
    opt[String]("copymodel")
      .text("if copy model default is false")
      .action((x,c) => c.copy(copymodel = x))
    opt[String]("taskdelstrategy")
      .text("task level del strategy")
      .action((x,c) => c.copy(taskdelstrategy = x))
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
    opt[String]("featureNormalize")
      .text("if z-score features")
      .action((x,c) => c.copy(featureNormalize = x))
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
