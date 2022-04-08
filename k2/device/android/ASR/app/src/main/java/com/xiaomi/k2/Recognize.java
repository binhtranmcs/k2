package com.xiaomi.k2;

public class Recognize {

  static {
    System.loadLibrary("k2");
  }

  public static void init(String modelPath, String dictPath) {}
  public static void reset() {}
  public static  void acceptWaveform(short[] waveform) {}
  public static  void setInputFinished() {}
  public static  boolean getFinished(){ return true;}
  public static  void startDecode(){}
  public static  String getResult(){ return "";}
}
