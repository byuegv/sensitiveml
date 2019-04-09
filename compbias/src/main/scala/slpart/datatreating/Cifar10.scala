package slpart.datatreating
import java.nio.ByteBuffer

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ArrayBuffer

object Cifar10 {
  val itemEachFile: Int = 10000
  val labelSize: Int = 1
  val featureSize: Int = 3072 //3*32*32
  val onePointSize: Int = labelSize + featureSize

  /**
    * 以统一的格式加载feature and label
    * @param filePath
    * @return
    */
  def loadData(filePath:String):ArrayBuffer[LabeledPoint] = {
    val bytes = BytesDataLoad.load(filePath)

    val labelArray = new ArrayBuffer[Int]
    val featureArray = new ArrayBuffer[Array[Int]]

    val newData = bytes.map(_ & 0xff)
    for(cur <- 0 until itemEachFile){
      labelArray += newData(cur*onePointSize) + 1
      featureArray += newData.slice(cur*onePointSize + labelSize,cur*onePointSize + labelSize + featureSize)
    }

    val pointArray = new ArrayBuffer[LabeledPoint]
    for(cur <- 0 until(itemEachFile)){
      pointArray += LabeledPoint(labelArray(cur).toDouble,Vectors.dense(featureArray(cur).map(_.toDouble)))
    }
    pointArray
  }

  /**
    * 加载cifa10的训练集
    * @param trainFilePath
    * @return
    */
  def loadTrainData(trainFilePath:String*):Array[LabeledPoint] = {
    val trainData = new ArrayBuffer[LabeledPoint]
    for(trainFile <- trainFilePath){
      trainData ++= loadData(trainFile)
    }
    trainData.toArray
  }

  /**
    * 加载cifar10的测试集
    * @param testFilePath
    * @return
    */
  def loadTestData(testFilePath:String):Array[LabeledPoint] = {
    loadData(testFilePath).toArray
  }
}
