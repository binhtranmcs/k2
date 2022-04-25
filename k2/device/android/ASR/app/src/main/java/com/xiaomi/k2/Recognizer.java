package com.xiaomi.k2;

public class Recognizer {

  static {
    System.loadLibrary("k2");
  }

  public static native void init(String modelPath, String bpePath);
  public static native String decode(float[] waveform);
}
