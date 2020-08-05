/**
 * Core implementation for RNN-based Magenta sketch models such as SketchRNN.
 *
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Imports
 */
import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
//import * as tfc from '@tensorflow/tfjs-converter';
import * as support from '../core/sketch_support';

/**
 * Interface for JSON specification of a `MusicVAE` model.
 *
 * @property max_seq_len: Model trained on dataset w/ this max sequence length.
 * @property mode: Pre-trained models have this parameter for legacy reasons.
 * 0 for VAE, 1 for Decoder only. This model is Decoder only (not used).
 * @property name: QuickDraw name, like cat, dog, elephant, etc
 * @property scale_factor: the factor to convert from neural-network space to
 * pixel space. Most pre-trained models have this number between 80-120
 * @property version: Pre-trained models have a version between 1-6, for
 * the purpose of experimental research log.
 */
export interface SketchRNNInfo {
  max_seq_len: number;
  mode: number;
  name: string;
  scale_factor: number;
  version: number;
}

/**
 * Interface for specification of the Probability Distribution Function
 * of a pen stroke.
 * 
 * Please refer to "A Neural Representation of Sketch Drawings"
 * https://arxiv.org/abs/1704.03477
 * 
 * In Eq.3 is an explanation of all of these parameters.
 * 
 * Below is a brief description:
 * 
 * @property pi: categorial distribution for mixture of Gaussian
 * @property muX: mean for x-axis
 * @property muY: mean for y-axis
 * @property sigmaX: standard deviation of x-axis
 * @property sigmaY: standard deviation of y-axis
 * @property corr: correlation parameter between x and y
 * @property pen: categorical distribution for the 3 pen states
 */
export interface StrokePDF {
  pi: Float32Array;
  muX: Float32Array;
  muY: Float32Array;
  sigmaX: Float32Array;
  sigmaY: Float32Array;
  corr: Float32Array;
  pen: Float32Array;
}

/**
 * States of the LSTM Cell
 * 
 * Long-Short Term Memory: ftp://ftp.idsia.ch/pub/juergen/lstm.pdf
 * 
 * @property c: memory "cell" of the LSTM.
 * @property h: hidden state (also the output) of the LSTM.
 */
export interface LSTMState {
  c: Float32Array;
  h: Float32Array;
}

/**
 * Main SketchRNN model class.
 *
 * Implementation of decoder model in https://arxiv.org/abs/1704.03477
 * 
 * TODO(hardmaru): make a "batch" continueSequence-like method
 * that runs fully on GPU.
 */
export class SketchRNN {
  private checkpointURL: string;
  private initialized: boolean;

  public info: SketchRNNInfo;
  public numUnits: number;

  public pixelFactor: number;
  public scaleFactor: number;

  public imgArray: tf.Tensor;

  public encoder: tfl.LayersModel;
  public decoder: tfl.LayersModel;
  public input_to_embed_dense: tfl.LayersModel;

  public wq_dense: tfl.LayersModel[] = new Array(4);
  public wk_dense: tfl.LayersModel[] = new Array(4);
  public wv_dense: tfl.LayersModel[] = new Array(4);
  public concat_attention_to_decoder_output: tfl.LayersModel[] = new Array(4);


  //public z_to_init: tfl.LayersModel;
  public output_model: tfl.LayersModel;




  public latentZ: tf.Tensor;
  private NMIXTURE = 20;
  public forgetBias: tf.Scalar;
   // raw weights and dimensions directly from JSON
  public weights: Float32Array[];
  public weightDims: number[][];
  // TensorFlow.js weight matrices
  public outputKernel: tf.Tensor2D;
  public outputBias: tf.Tensor1D;
  public lstmKernel: tf.Tensor2D;
  public lstmBias: tf.Tensor1D;
  public rawVars: tfl.LayersModel;
  /**f
   * `SketchRNN` constructor.
   *
   * @param checkpointURL Path to the checkpoint directory.
   */
  constructor(checkpointURL: string) {
    console.log('construct model');
    this.checkpointURL = checkpointURL;
    this.initialized = false;
    console.log(this.checkpointURL);
  }

  /**
   * Returns true if model is intialized.
   */
  isInitialized() {
    return this.initialized;
  }

  /**
   * Instantiates class information inputs from the JSON model.
   * TODO(hardmaru): to add support for new tfjs checkpoints.
   */
  private instantiateFromJSON(info: SketchRNNInfo) {

    this.info = info;
    this.setPixelFactor(2.0);

  }
  




  /**
   * Loads variables from the JSON model
   */
  async initialize() {
    this.dispose();
    console.log(this.checkpointURL); 
    const vars = await fetch(this.checkpointURL.concat('/Info.json'))
      .then((response) => response.json());





    this.instantiateFromJSON(vars[0]);
    this.encoder = await tfl.loadLayersModel(this.checkpointURL.concat('/encoder_model_without_highpass/model.json'));
    console.log('Initialized encoder!!!');

    this.output_model = await tfl.loadLayersModel(this.checkpointURL.concat('/output_model/model.json'));
    console.log('Initialized output_model');
    
    this.input_to_embed_dense = await tfl.loadLayersModel(this.checkpointURL.concat('/input_to_embed_dense/model.json'));
    console.log('Initialized input_to_embed_dense');

    
    
    
    for (let i = 0; i < 4; i ++){
      this.wq_dense[i] = await tfl.loadLayersModel(this.checkpointURL+'/wq_dense_'+i+'/model.json');
      console.log('Initialized wq_dense_'+i);

      this.wk_dense[i] = await tfl.loadLayersModel(this.checkpointURL+'/wk_dense_'+i+'/model.json');
      console.log('Initialized wk_dense_'+i);

      this.wv_dense[i] = await tfl.loadLayersModel(this.checkpointURL+'/wv_dense_'+i+'/model.json');
      console.log('Initialized wv_dense_'+i);

      this.concat_attention_to_decoder_output[i] = await tfl.loadLayersModel(this.checkpointURL+'/concat_attention_to_decoder_output_'+i+'_model/model.json');
      console.log('Initialized concat_attention_to_decoder_output_'+i);
    }
   

   


    //var lstm = this.decoder.getLayer('lstm');

    
    //this.numUnits = lstm.getWeights()[1].shape[0]; // size of LSTM
  
    var imgDataReq = await fetch(this.checkpointURL.concat('/image/0.json'));
    this.imgArray = await imgDataReq.json();
    this.initialized = true;
    var a: any[] = Array.from([1,2,3]);
    console.log(a);
    console.log('Initialized SketchRNN.');
  }

  dispose() {
    if (this.encoder) {
      this.encoder.dispose();
      this.encoder = undefined;
      console.log('disposed encoder.');
    }

    if (this.output_model) {
       this.output_model.dispose();
       this.output_model = undefined;
       console.log('disposed output_model');
    }

    if (this.input_to_embed_dense) {
       this.input_to_embed_dense.dispose();
       this.input_to_embed_dense = undefined;
       console.log('disposed input_to_embed_dense');
    }
    
    for (let i = 0; i < 4; i ++){
      if (this.wq_dense[i]) {
       this.wq_dense[i].dispose();
       this.wq_dense[i] = undefined;
       console.log('disposed wq_dense_'+i);
      }
      if (this.wk_dense[i]) {
       this.wk_dense[i].dispose();
       this.wk_dense[i] = undefined;
       console.log('disposed wk_dense_'+i);
      }
      if (this.wv_dense[i]) {
       this.wv_dense[i].dispose();
       this.wv_dense[i] = undefined;
       console.log('disposed wv_dense_'+i);
      }
      if (this.concat_attention_to_decoder_output[i]) {
       this.concat_attention_to_decoder_output[i].dispose();
       this.concat_attention_to_decoder_output[i] = undefined;
       console.log('disposed concat_attention_to_decoder_output_'+i);
      }
    }
    
    this.initialized = false;
  }

 /**
   * Sets the internal EXTRA factor of this model (pixel to model space)
   *
   * @param scale (the extra scale factor for pixel to model space)
   *
   * @returns nothing
   */
  setPixelFactor(scale: number) {
    // for best effect, set to 1.0 for d3 or paper.js, 2.0 for p5.js
    this.pixelFactor = scale;
    this.scaleFactor = this.info.scale_factor / this.pixelFactor;
  }


  /**
   * Returns the zero/initial state of the model
   *
   * @returns zero state of the lstm: [c, h], where c and h are zero vectors.
   */
  zeroState() {
    const result:LSTMState = {
      c: new Float32Array(this.numUnits),
      h: new Float32Array(this.numUnits)
    };
    return result;
  }
 
  /**
   * Given the RNN state, returns the probability distribution function (pdf)
   * of the next stroke. Optionally adjust the temperature of the pdf here.
   *
   * @param state previous LSTMState.
   * @param temperature (Optional) for dx and dy (default 0.65)
   * @param softmaxTemperature (Optional) for Pi and Pen discrete states
   * (default is temperature * 0.5 + 0.5, which is a nice heuristic.)
   *
   * @returns StrokePDF (pi, muX, muY, sigmaX, sigmaY, corr, pen)
   */
  getPDF(logit:tf.Tensor,
    temperature=0.65,
    softmaxTemperature?: number) {
    const temp = temperature;
    let discreteTemp: number = 0.5 + temp * 0.5; // good heuristic.
    if (softmaxTemperature) {
      discreteTemp = softmaxTemperature;
    }
    const NOUT = this.NMIXTURE;
    const out = tf.tidy(() => {

      const sqrttemp = tf.scalar(Math.sqrt(temp));
      const softtemp = tf.scalar(discreteTemp);

      const z = logit;
      const [rawPen, rst] = tf.split(z, [3, NOUT*6], 1);
      const [rawPi, mu1, mu2, rawSigma1, rawSigma2, rawCorr] = tf.split(rst, 6, 1);
      const pen = tf.softmax(rawPen.div(softtemp));
      const pi = tf.softmax(rawPi.div(softtemp));
      const sigma1 = tf.exp(rawSigma1).mul(sqrttemp);
      const sigma2 = tf.exp(rawSigma2).mul(sqrttemp);
      const corr = tf.tanh(rawCorr);
      const result = [pi, mu1, mu2, sigma1, sigma2, corr, pen];
      // concat, and then unpack after dataSync
      return tf.concat(result,1);
    });
    const result = out.dataSync();
    out.dispose();
    const pdf:StrokePDF = { // note: JS doesn't have a nice "split" method.
      pi: new Float32Array(result.slice(0, NOUT)),
      muX: new Float32Array(result.slice(1*NOUT, 2*NOUT)),
      muY: new Float32Array(result.slice(2*NOUT, 3*NOUT)),
      sigmaX: new Float32Array(result.slice(3*NOUT, 4*NOUT)),
      sigmaY: new Float32Array(result.slice(4*NOUT, 15*NOUT)),
      corr: new Float32Array(result.slice(5*NOUT, 6*NOUT)),
      pen: new Float32Array(result.slice(6*NOUT, 6*NOUT+3))
    };
    return pdf;
  }

  /**
   * Returns a new copy of the rnn state
   *
   * @param rnnState original LSTMState
   *
   * @returns copy of LSTMState
   */
  copyState(rnnState: LSTMState) {
    const result:LSTMState = {
      c: new Float32Array(rnnState.c),
      h: new Float32Array(rnnState.h)
    };
    return result;
  }

  /**
   * Returns the zero input state of the model
   *
   * @returns [0, 0, 1, 0, 0].
   */
  zeroInput() {
    return [0, 0, 1, 0, 0];
  }
 
  getNMIXTURE() {
    return this.NMIXTURE;
}

  /**
   * Samples the next point of the sketch given pdf parameters
   *
   * @param pdf result from getPDF() call
   *
   * @returns [dx, dy, penDown, penUp, penEnd]
   */
  sample(pdf: StrokePDF) {
    // pdf is a StrokePDF Interface
    // returns [x, y, eos]
    console.log(pdf.pi);
    const idx = support.sampleSoftmax(pdf.pi);
    const mu1 = pdf.muX[idx];
    const mu2 = pdf.muY[idx];
    const sigma1 = pdf.sigmaX[idx];
    const sigma2 = pdf.sigmaY[idx];
    const corr = pdf.corr[idx];
    const penIdx = support.sampleSoftmax(pdf.pen);
    const penstate = [0, 0, 0];
    penstate[penIdx] = 1;
    const delta = support.birandn(mu1, mu2, sigma1, sigma2, corr);
    const stroke = [
      delta[0] * this.scaleFactor,
      delta[1] * this.scaleFactor,
      penstate[0],
      penstate[1],
      penstate[2]
    ];
    return stroke;
  }

  /**
   * Simplifies line using RDP algorithm
   *
   * @param line list of points [[x0, y0], [x1, y1], ...]
   * @param tolerance (Optional) default 2.0
   *
   * @returns simpified line [[x0', y0'], [x1', y1'], ...]
   */
  simplifyLine(line: number[][], tolerance?: number) {
    return support.simplifyLine(line, tolerance);
  }

  /**
   * Simplifies lines using RDP algorithm
   *
   * @param line list of lines (each element is [[x0, y0], [x1, y1], ...])
   * @param tolerance (Optional) default 2.0
   *
   * @returns simpified lines (each elem is [[x0', y0'], [x1', y1'], ...])
   */
  simplifyLines(lines: number[][][], tolerance?: number) {
    return support.simplifyLines(lines, tolerance);
  }

  /**
   * Convert from polylines to stroke-5 format that sketch-rnn uses
   *
   * @param lines list of points each elem is ([[x0, y0], [x1, y1], ...])
   *
   * @returns stroke-5 format of the line, list of [dx, dy, p0, p1, p2]
   */
  linesToStroke(lines: number[][][]) {
    return support.linesToStrokes(lines);
  }

  /**
   * Convert from a line format to stroke-5
   *
   * @param line list of points [[x0, y0], [x1, y1], ...]
   * @param lastPoint the absolute position of the last point
   *
   * @returns stroke-5 format of the line, list of [dx, dy, p0, p1, p2]
   */
  lineToStroke(line: number[][], lastPoint: number[]) {
    return support.lineToStroke(line, lastPoint);
  }

}
