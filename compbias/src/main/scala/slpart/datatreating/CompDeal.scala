package slpart.datatreating
import AccurateML.nonLinearRegression.ZFHash3
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
  * 用于粗粒度压缩点的生成
  */
object CompDeal {
  val compScale: Double = 1000

  /**
    * 生成粗粒度压缩点
    * 并将压缩点和原始点组合成一个新的sample
    *
    * @param trainLabeledPoint
    * @param itqbitN
    * @param itqitN    压缩算法迭代次数,number of loops to train incremental SVD
    * @param itqratioN 压缩算法每itqratioN个属性中使用一个属性进行SVD,the number of features used to train SVD is "numFeature/itqitN"
    * @param upBound   每个压缩点包含原始点个数的上限
    * @param splitN
    * @param isSparse  The input data set is sparse ("true") or dense ("false")
    * @return
    */
  def getCompTrainData(trainLabeledPoint: RDD[LabeledPoint], itqbitN: Int = 1,
                       itqitN: Int = 20,
                       itqratioN: Int = 100,
                       upBound: Int = 20,
                       splitN: Double = 2.0,
                       isSparse: Boolean = false) = {
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse)
    val objectData = trainLabeledPoint.mapPartitions(oHash.zfHashMap).map(_._1)
    //增加ID
    val sap = objectData.zipWithIndex().map(p => {
      val zipAndOrig = p._1
      val zipAry = zipAndOrig.head
      val origAry = zipAndOrig.last
      val compId = p._2 + 2

      val feaAry = new ArrayBuffer[Double]
      val labAry = new ArrayBuffer[Double]

      val compfea = zipAry(0).features.toArray
      //      val complab = zipAry(0).label + compId * compScale
      val complab = zipAry(0).label

      feaAry ++= compfea
      labAry += complab

      for (elem <- origAry) {
        val origfea = elem.features.toArray
        val origlab = elem.label
        feaAry ++= origfea
        labAry += origlab
      }
      (feaAry, labAry)
    })
    val oneSample = sap.first()
    val singleFeatureLength = oneSample._1.length / oneSample._2.length
    val comp2Orig = sap.map(_._2.length).max()
    println(s"comp-origs:1 - ${comp2Orig - 1} origFeatureSize:${singleFeatureLength} combineFeatureSize:${singleFeatureLength * comp2Orig}")

    //对其压缩点和原始点的对应
    val padSap = sap.map(p => {
      val feaAry = p._1
      val labAry = p._2
      val preSize = labAry.length
      val padFea = feaAry.slice(singleFeatureLength + 1, singleFeatureLength + singleFeatureLength + 1)
      val padLab = labAry(1)

      var curSize = preSize
      while (curSize < comp2Orig) {
        feaAry ++= padFea
        labAry += padLab
        curSize += 1
      }
      Sample(Tensor(T(feaAry.head, feaAry.tail: _*)), Tensor(T(labAry.head, labAry.tail: _*)))
    })
    padSap
  }

  def getCompTrainDataPartition(trainLabeledPoint: RDD[LabeledPoint], itqbitN: Int = 1,
                                itqitN: Int = 20,
                                itqratioN: Int = 100,
                                upBound: Int = 20,
                                splitN: Double = 2.0,
                                isSparse: Boolean = false) = {
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse)
    val objectData = trainLabeledPoint.mapPartitions(oHash.zfHashMap).map(_._1)
    //增加ID
    val sap = objectData.zipWithIndex().mapPartitions(p => new CustomZipAryIterator[Float](p))
    val oneSample = sap.first()
    val singleFeatureLength = oneSample._1.length / oneSample._2.length
    val comp2Orig = sap.mapPartitions(iter => Iterator.single(iter.next()._2.length)).max()
    println(s"comp-origs:1 - ${comp2Orig - 1} origFeatureSize:${singleFeatureLength} combineFeatureSize:${singleFeatureLength * comp2Orig}")

    //对其压缩点和原始点的对应
    val padSap = sap.mapPartitions(p => new CustomZipSampleIterator[Float](p,singleFeatureLength,comp2Orig))
    padSap
  }
  def getCompTrainDataZip(trainLabeledPoint: RDD[LabeledPoint], itqbitN: Int = 1,
                       itqitN: Int = 20,
                       itqratioN: Int = 100,
                       upBound: Int = 20,
                       splitN: Double = 2.0,
                       isSparse: Boolean = false) = {
    val oHash = new ZFHash3(itqbitN, itqitN, itqratioN, upBound, splitN, isSparse)
    val objectData = trainLabeledPoint.mapPartitions(oHash.zfHashMap).map(_._1)
    //增加ID
    val sap = objectData.zipWithIndex().map(p => {
      val zipAndOrig = p._1
      val zipAry = zipAndOrig.head

      val compfea = zipAry.head.features.toArray
      // val complab = zipAry(0).label + compId * compScale
      val complab = zipAry.head.label

      Sample(Tensor(T(compfea.head,compfea.tail: _*)),Tensor(T(complab)))
    })
   sap
  }
}


