package AccurateML.svm.shun.v1

/**
  * ICT
  * Created by douyishun on 11/23/16.
  */

import java.io.FileWriter

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

object SVMExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("SVMKernel")
    val sc = new SparkContext(conf)
    //    val dataSet: String = "heart"
    val initW: Double = args(0).toDouble
    val step: Double = args(1).toDouble
    val numFeature: Int = args(2).toInt //10
    val ineritN: Int = args(3).toInt //5
    val itN: Int = args(4).toInt
    val testPath: String = args(5)
    val dataPath: String = args(6)
    val test100: Boolean = args(7).toBoolean
    val weightsPath = args(8)
    val minPartN = args(9).toInt

    val rDivN = 100
    //    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100)
    val ratioL = if (test100) List(rDivN) else List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)

    val mesb = new ArrayBuffer[Double]()
    val nntimesb = new ArrayBuffer[Double]()
    val traintsb = new ArrayBuffer[Double]()

    println(this.getClass.getName + ",data," + dataPath + ",test," + testPath)

    for (r <- ratioL) {
      val ratio = r / rDivN.toDouble

      val training = MLUtils.loadLibSVMFile(sc, dataPath, numFeature, minPartN).sample(false, ratio) //"data/" + dataSet + "/train"
      val testing = MLUtils.loadLibSVMFile(sc, testPath, numFeature, minPartN) //"data/" + dataSet + "/test"

      val t1 = System.currentTimeMillis
      val model = new SVM(training).train(
        maxLayer = itN, //10
        maxIterations = ineritN, //1000
        C = 1,
        gamma = 0.05)
      val t2 = System.currentTimeMillis

      val scoreAndLabels1 = Scale.scale(training).map { point =>
        val score = model.predictPoint(point.features)
        (score, point.label.toInt)
      }

      val scoreAndLabels2 = Scale.scale(testing).map { point =>
        val score = model.predictPoint(point.features)
        (score, point.label.toInt)
      }
      val fw = new FileWriter(weightsPath, true)
      fw.write("Ratio," + ratio + ",TrainACC," + model.modelAccuracy(scoreAndLabels1) + ",TestACC," + model.modelAccuracy(scoreAndLabels2) + ",TrainT, " + (t2 - t1).toDouble + ",maxLayer," + itN + "\n")
      fw.close()
      println("Ratio," + ratio + ",TrainACC," + model.modelAccuracy(scoreAndLabels1) + ",TestACC," + model.modelAccuracy(scoreAndLabels2) + ",TrainT, " + (t2 - t1).toDouble + ",maxLayer," + itN)

    }


  }
}
