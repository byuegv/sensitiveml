package slpart.datatreating
import com.intel.analytics.bigdl.dataset.ByteRecord
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.utils.File

import scala.collection.mutable.ArrayBuffer
object Cifar10ByteRecord {
  val hdfsPrefix = "hdfs:"


  def loadTrain(dataFile:String*): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    dataFile.foreach(load(_,result))
    result.toArray
  }

  def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    load(dataFile,result)
    result.toArray
  }


  def load(featureFile: String, result: ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1
    val channelOffset = rowNum * colNum
    val bufferOffset = 8

    val featureBuffer = if(featureFile.startsWith(hdfsPrefix)){
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    }
    else{
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }

    val featureArray = featureBuffer.array()
    val featureCount  = featureArray.length / (rowNum*colNum * 3 +1)

    var i = 0
    while(i < featureCount){
      val img = new Array[Byte](rowNum*colNum*3+bufferOffset)
      val byteBuffer = ByteBuffer.wrap(img)

      byteBuffer.putInt(rowNum)
      byteBuffer.putInt(colNum)

      val label = featureArray(i*imageOffset).toFloat
      var y = 0
      val start = i*imageOffset + 1
      while(y < rowNum){
        var x = 0
        while(x < colNum){
          img((x+y*colNum)*3+2+bufferOffset) = featureArray(start+x+y*colNum)
          img((x+y*colNum)*3+1+bufferOffset) = featureArray(start+x+y*colNum+channelOffset)
          img((x+y*colNum)*3 + bufferOffset) = featureArray(start+x+y*colNum+2*channelOffset)
          x += 1
        }
        y += 1
      }
      result.append(ByteRecord(img,label + 1.0f))
      i += 1
    }
  }
}
