package com.intel.analytics.bigdl.optim

import scala.collection.mutable.ArrayBuffer

/**
  * task级别以及epoch级别
  * 压缩点的删除策略
  */
object FilterStrategy {
  /**
    * task级别的删除策略
    * @param gradAry 梯度信息Array[(compId,index,gradinfo)]
    * @param taskdelratio task 级别的删除比例
    * @param taskdropdecay task 级别不计入删除的比例
    * @return
    */
  def taskFilter(gradAry: Array[(Int,Int,Double)],
                 taskdelratio: Double = 0.01,
                 taskdropdecay: Double = 0.0) = {
    //按照梯度信息从小到大排序
    val sortedGradAry = gradAry.sortWith(_._3 < _._3)
    val n = sortedGradAry.length
    val dropN = (n * taskdropdecay + 0.5).toInt

    var areaSum: Double = 0.0
    for(idx <- 0 until(n - dropN)){
      areaSum += sortedGradAry(idx)._3
    }
    var curArea: Double = 0.0
    val selectAry = new ArrayBuffer[(Int,Int)]
    val deleteAry = new ArrayBuffer[(Int,Int)]
    var cur: Int = 0
    while(cur < n){
      curArea += sortedGradAry(cur)._3
      if(curArea >= areaSum * taskdelratio){
        selectAry += Tuple2(sortedGradAry(cur)._1,sortedGradAry(cur)._2)
      }
      else{
        deleteAry += Tuple2(sortedGradAry(cur)._1,sortedGradAry(cur)._2)
      }
      cur += 1
    }
    (selectAry,deleteAry)
  }
  def taskFilterIndex(gradAry: Array[(Int,Double)],taskdelratio: Double =0.01,taskdropdecay:Double = 0.0) ={
    val sortedGradAry = gradAry.sortWith(_._2 < _._2)
    val n = sortedGradAry.length
    val dropN = (n * taskdropdecay + 0.5).toInt

    var areaSum: Double = 0.0
    for(idx <- 0 until(n - dropN)){
      areaSum += sortedGradAry(idx)._2
    }
    var curArea: Double = 0.0
    val selectAry = new ArrayBuffer[Int]
    val deleteAry = new ArrayBuffer[Int]
    var cur: Int = 0
    while(cur < n){
      curArea += sortedGradAry(cur)._2
      if(curArea >= areaSum * taskdelratio){
        selectAry += sortedGradAry(cur)._1
      }
      else{
        deleteAry += sortedGradAry(cur)._1
      }
      cur += 1
    }
    (selectAry,deleteAry)
  }
}
