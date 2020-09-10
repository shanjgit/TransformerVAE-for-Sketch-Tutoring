import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
export interface SketchRNNInfo {
    max_seq_len: number;
    mode: number;
    name: string;
    scale_factor: number;
    version: number;
}
export interface StrokePDF {
    pi: Float32Array;
    muX: Float32Array;
    muY: Float32Array;
    sigmaX: Float32Array;
    sigmaY: Float32Array;
    corr: Float32Array;
    pen: Float32Array;
}
export interface LSTMState {
    c: Float32Array;
    h: Float32Array;
}
export declare class SketchRNN {
    private checkpointURL;
    private initialized;
    info: SketchRNNInfo;
    numUnits: number;
    pixelFactor: number;
    scaleFactor: number;
    imgArray: tf.Tensor;
    encoder: tfl.LayersModel;
    decoder: tfl.LayersModel;
    input_to_embed_dense: tfl.LayersModel;
    wq_dense: tfl.LayersModel[];
    wk_dense: tfl.LayersModel[];
    wv_dense: tfl.LayersModel[];
    concat_attention_to_decoder_output: tfl.LayersModel[];
    output_model: tfl.LayersModel;
    latentZ: tf.Tensor;
    private NMIXTURE;
    forgetBias: tf.Scalar;
    weights: Float32Array[];
    weightDims: number[][];
    outputKernel: tf.Tensor2D;
    outputBias: tf.Tensor1D;
    lstmKernel: tf.Tensor2D;
    lstmBias: tf.Tensor1D;
    rawVars: tfl.LayersModel;
    constructor(checkpointURL: string);
    isInitialized(): boolean;
    private instantiateFromJSON;
    initialize(): Promise<void>;
    dispose(): void;
    setPixelFactor(scale: number): void;
    zeroState(): LSTMState;
    getPDF(logit: tf.Tensor, temperature?: number, softmaxTemperature?: number): StrokePDF;
    copyState(rnnState: LSTMState): LSTMState;
    zeroInput(): number[];
    getNMIXTURE(): number;
    sample(pdf: StrokePDF): number[];
    simplifyLine(line: number[][], tolerance?: number): number[][];
    simplifyLines(lines: number[][][], tolerance?: number): number[][][];
    linesToStroke(lines: number[][][]): number[][];
    lineToStroke(line: number[][], lastPoint: number[]): number[][];
}
