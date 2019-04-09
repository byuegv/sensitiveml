package slpart.datatreating

import com.intel.analytics.bigdl.dataset.ByteRecord
import java.nio.ByteBuffer
import java.nio.file.{Files,Path,Paths}
import com.intel.analytics.bigdl.utils.File

object MnistByteRecord {
  val hdfsPrefix = "hdfs:"
  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {
    val featureBuffer = if(featureFile.startsWith(hdfsPrefix)){
      ByteBuffer.wrap(File.readHdfsByte((featureFile)))
    }
    else{
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if(featureFile.startsWith(hdfsPrefix)){
      ByteBuffer.wrap(File.readHdfsByte(labelFile))
    }
    else{
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }

    val labelMagicNumber = labelBuffer.getInt()
    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while(i < featureCount){
      val img = new Array[Byte]((rowNum*colNum))
      var y = 0
      while(y < rowNum){
        var x = 0
        while(x < colNum){
          img(x + y*colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img,labelBuffer.get().toFloat + 1.0f)
      i += 1
    }
    result
  }

}
