package com.intel.analytics.bigdl.optim

object CompUtils {
  val compScale: Double = 1000d
  def paraseIdLabel(idLabel: Double) = {
    val id = (idLabel / compScale + 0.5).toInt
    val label = idLabel - id * compScale
    (id,label)
  }

}
