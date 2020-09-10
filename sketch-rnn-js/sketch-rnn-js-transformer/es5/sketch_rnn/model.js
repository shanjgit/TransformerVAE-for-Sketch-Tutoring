"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
var tfl = require("@tensorflow/tfjs-layers");
var support = require("../core/sketch_support");
var SketchRNN = (function () {
    function SketchRNN(checkpointURL) {
        this.wq_dense = new Array(4);
        this.wk_dense = new Array(4);
        this.wv_dense = new Array(4);
        this.concat_attention_to_decoder_output = new Array(4);
        this.NMIXTURE = 20;
        console.log('construct model');
        this.checkpointURL = checkpointURL;
        this.initialized = false;
        console.log(this.checkpointURL);
    }
    SketchRNN.prototype.isInitialized = function () {
        return this.initialized;
    };
    SketchRNN.prototype.instantiateFromJSON = function (info) {
        this.info = info;
        this.setPixelFactor(2.0);
    };
    SketchRNN.prototype.initialize = function () {
        return __awaiter(this, void 0, void 0, function () {
            var vars, _a, _b, _c, i, _d, _e, _f, _g, _h, _j, _k, _l, imgDataReq, _m, a;
            return __generator(this, function (_o) {
                switch (_o.label) {
                    case 0:
                        this.dispose();
                        console.log(this.checkpointURL);
                        return [4, fetch(this.checkpointURL.concat('/Info.json'))
                                .then(function (response) { return response.json(); })];
                    case 1:
                        vars = _o.sent();
                        this.instantiateFromJSON(vars[0]);
                        _a = this;
                        return [4, tfl.loadLayersModel(this.checkpointURL.concat('/encoder_model_without_highpass_model/model.json'))];
                    case 2:
                        _a.encoder = _o.sent();
                        console.log('Initialized encoder!!!');
                        _b = this;
                        return [4, tfl.loadLayersModel(this.checkpointURL.concat('/output_model/model.json'))];
                    case 3:
                        _b.output_model = _o.sent();
                        console.log('Initialized output_model');
                        _c = this;
                        return [4, tfl.loadLayersModel(this.checkpointURL.concat('/input_to_embed_dense/model.json'))];
                    case 4:
                        _c.input_to_embed_dense = _o.sent();
                        console.log('Initialized input_to_embed_dense');
                        i = 0;
                        _o.label = 5;
                    case 5:
                        if (!(i < 4)) return [3, 11];
                        _d = this.wq_dense;
                        _e = i;
                        return [4, tfl.loadLayersModel(this.checkpointURL + '/wq_dense_' + i + '/model.json')];
                    case 6:
                        _d[_e] = _o.sent();
                        console.log('Initialized wq_dense_' + i);
                        _f = this.wk_dense;
                        _g = i;
                        return [4, tfl.loadLayersModel(this.checkpointURL + '/wk_dense_' + i + '/model.json')];
                    case 7:
                        _f[_g] = _o.sent();
                        console.log('Initialized wk_dense_' + i);
                        _h = this.wv_dense;
                        _j = i;
                        return [4, tfl.loadLayersModel(this.checkpointURL + '/wv_dense_' + i + '/model.json')];
                    case 8:
                        _h[_j] = _o.sent();
                        console.log('Initialized wv_dense_' + i);
                        _k = this.concat_attention_to_decoder_output;
                        _l = i;
                        return [4, tfl.loadLayersModel(this.checkpointURL + '/concat_attention_to_decoder_output_' + i + '_model_try/model.json')];
                    case 9:
                        _k[_l] = _o.sent();
                        console.log('Initialized concat_attention_to_decoder_output_' + i);
                        _o.label = 10;
                    case 10:
                        i++;
                        return [3, 5];
                    case 11: return [4, fetch(this.checkpointURL.concat('/image/0.json'))];
                    case 12:
                        imgDataReq = _o.sent();
                        _m = this;
                        return [4, imgDataReq.json()];
                    case 13:
                        _m.imgArray = _o.sent();
                        this.initialized = true;
                        a = Array.from([1, 2, 3]);
                        console.log(a);
                        console.log('Initialized SketchRNN.');
                        return [2];
                }
            });
        });
    };
    SketchRNN.prototype.dispose = function () {
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
        for (var i = 0; i < 4; i++) {
            if (this.wq_dense[i]) {
                this.wq_dense[i].dispose();
                this.wq_dense[i] = undefined;
                console.log('disposed wq_dense_' + i);
            }
            if (this.wk_dense[i]) {
                this.wk_dense[i].dispose();
                this.wk_dense[i] = undefined;
                console.log('disposed wk_dense_' + i);
            }
            if (this.wv_dense[i]) {
                this.wv_dense[i].dispose();
                this.wv_dense[i] = undefined;
                console.log('disposed wv_dense_' + i);
            }
            if (this.concat_attention_to_decoder_output[i]) {
                this.concat_attention_to_decoder_output[i].dispose();
                this.concat_attention_to_decoder_output[i] = undefined;
                console.log('disposed concat_attention_to_decoder_output_' + i);
            }
        }
        this.initialized = false;
    };
    SketchRNN.prototype.setPixelFactor = function (scale) {
        this.pixelFactor = scale;
        this.scaleFactor = this.info.scale_factor / this.pixelFactor;
    };
    SketchRNN.prototype.zeroState = function () {
        var result = {
            c: new Float32Array(this.numUnits),
            h: new Float32Array(this.numUnits)
        };
        return result;
    };
    SketchRNN.prototype.getPDF = function (logit, temperature, softmaxTemperature) {
        if (temperature === void 0) { temperature = 0.65; }
        var temp = temperature;
        var discreteTemp = 0.5 + temp * 0.5;
        if (softmaxTemperature) {
            discreteTemp = softmaxTemperature;
        }
        var NOUT = this.NMIXTURE;
        var out = tf.tidy(function () {
            var sqrttemp = tf.scalar(Math.sqrt(temp));
            var softtemp = tf.scalar(discreteTemp);
            var z = logit;
            var _a = tf.split(z, [3, NOUT * 6], 1), rawPen = _a[0], rst = _a[1];
            var _b = tf.split(rst, 6, 1), rawPi = _b[0], mu1 = _b[1], mu2 = _b[2], rawSigma1 = _b[3], rawSigma2 = _b[4], rawCorr = _b[5];
            var pen = tf.softmax(rawPen.div(softtemp));
            var pi = tf.softmax(rawPi.div(softtemp));
            var sigma1 = tf.exp(rawSigma1).mul(sqrttemp);
            var sigma2 = tf.exp(rawSigma2).mul(sqrttemp);
            var corr = tf.tanh(rawCorr);
            var result = [pi, mu1, mu2, sigma1, sigma2, corr, pen];
            return tf.concat(result, 1);
        });
        var result = out.dataSync();
        out.dispose();
        var pdf = {
            pi: new Float32Array(result.slice(0, NOUT)),
            muX: new Float32Array(result.slice(1 * NOUT, 2 * NOUT)),
            muY: new Float32Array(result.slice(2 * NOUT, 3 * NOUT)),
            sigmaX: new Float32Array(result.slice(3 * NOUT, 4 * NOUT)),
            sigmaY: new Float32Array(result.slice(4 * NOUT, 15 * NOUT)),
            corr: new Float32Array(result.slice(5 * NOUT, 6 * NOUT)),
            pen: new Float32Array(result.slice(6 * NOUT, 6 * NOUT + 3))
        };
        return pdf;
    };
    SketchRNN.prototype.copyState = function (rnnState) {
        var result = {
            c: new Float32Array(rnnState.c),
            h: new Float32Array(rnnState.h)
        };
        return result;
    };
    SketchRNN.prototype.zeroInput = function () {
        return [0, 0, 1, 0, 0];
    };
    SketchRNN.prototype.getNMIXTURE = function () {
        return this.NMIXTURE;
    };
    SketchRNN.prototype.sample = function (pdf) {
        console.log(pdf.pi);
        var idx = support.sampleSoftmax(pdf.pi);
        var mu1 = pdf.muX[idx];
        var mu2 = pdf.muY[idx];
        var sigma1 = pdf.sigmaX[idx];
        var sigma2 = pdf.sigmaY[idx];
        var corr = pdf.corr[idx];
        var penIdx = support.sampleSoftmax(pdf.pen);
        var penstate = [0, 0, 0];
        penstate[penIdx] = 1;
        var delta = support.birandn(mu1, mu2, sigma1, sigma2, corr);
        var stroke = [
            delta[0] * this.scaleFactor,
            delta[1] * this.scaleFactor,
            penstate[0],
            penstate[1],
            penstate[2]
        ];
        return stroke;
    };
    SketchRNN.prototype.simplifyLine = function (line, tolerance) {
        return support.simplifyLine(line, tolerance);
    };
    SketchRNN.prototype.simplifyLines = function (lines, tolerance) {
        return support.simplifyLines(lines, tolerance);
    };
    SketchRNN.prototype.linesToStroke = function (lines) {
        return support.linesToStrokes(lines);
    };
    SketchRNN.prototype.lineToStroke = function (line, lastPoint) {
        return support.lineToStroke(line, lastPoint);
    };
    return SketchRNN;
}());
exports.SketchRNN = SketchRNN;
//# sourceMappingURL=model.js.map