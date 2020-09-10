"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function stringToUint8array(b64encoded) {
    var u8 = new Uint8Array(atob(b64encoded)
        .split("")
        .map(function (c) {
        return c.charCodeAt(0);
    }));
    return u8;
}
exports.stringToUint8array = stringToUint8array;
function stringToArray(b64encoded) {
    var u = stringToUint8array(b64encoded);
    var result = new Int16Array(u.buffer);
    return result;
}
exports.stringToArray = stringToArray;
var returnV = false;
var vVal = 0.0;
function gaussRandom() {
    if (returnV) {
        returnV = false;
        return vVal;
    }
    var u = 2 * Math.random() - 1;
    var v = 2 * Math.random() - 1;
    var r = u * u + v * v;
    if (r === 0 || r > 1) {
        return gaussRandom();
    }
    var c = Math.sqrt(-2 * Math.log(r) / r);
    vVal = v * c;
    returnV = true;
    return u * c;
}
exports.gaussRandom = gaussRandom;
function randf(a, b) {
    return Math.random() * (b - a) + a;
}
exports.randf = randf;
function randi(a, b) {
    return Math.floor(Math.random() * (b - a) + a);
}
exports.randi = randi;
function randn(mu, std) {
    return mu + gaussRandom() * std;
}
exports.randn = randn;
function birandn(mu1, mu2, std1, std2, rho) {
    var z1 = randn(0, 1);
    var z2 = randn(0, 1);
    var x = Math.sqrt(1 - rho * rho) * std1 * z1 + rho * std1 * z2 + mu1;
    var y = std2 * z2 + mu2;
    return [x, y];
}
exports.birandn = birandn;
function sampleSoftmax(zSample) {
    var x = randf(0, 1);
    var N = zSample.length;
    var accumulate = 0;
    var i;
    for (i = 0; i < N; i++) {
        accumulate += zSample[i];
        if (accumulate >= x) {
            return i;
        }
    }
    console.log('error sampling pi index');
    return -1;
}
exports.sampleSoftmax = sampleSoftmax;
function simplifyLine(V, tolerance) {
    if (tolerance === void 0) { tolerance = 2.0; }
    var tol = tolerance;
    function diff(u, v) {
        return [u[0] - v[0], u[1] - v[1]];
    }
    function dot(u, v) {
        return u[0] * v[0] + u[1] * v[1];
    }
    function norm2(v) {
        return v[0] * v[0] + v[1] * v[1];
    }
    function d2(u, v) { return norm2(diff(u, v)); }
    function simplifyDP(tol, v, j, k, mk) {
        if (k <= j + 1) {
            return;
        }
        var maxi = j;
        var maxd2 = 0;
        var tol2 = tol * tol;
        var S = [v[j], v[k]];
        var u = diff(S[1], S[0]);
        var cu = norm2(u);
        var w;
        var pb;
        var b, cw, dv2;
        for (var i_1 = j + 1; i_1 < k; i_1++) {
            w = diff(v[i_1], S[0]);
            cw = dot(w, u);
            if (cw <= 0) {
                dv2 = d2(v[i_1], S[0]);
            }
            else if (cu <= cw) {
                dv2 = d2(v[i_1], S[1]);
            }
            else {
                b = cw / cu;
                pb = [S[0][0] + b * u[0], S[0][1] + b * u[1]];
                dv2 = d2(v[i_1], pb);
            }
            if (dv2 <= maxd2) {
                continue;
            }
            maxi = i_1;
            maxd2 = dv2;
        }
        if (maxd2 > tol2) {
            mk[maxi] = 1;
            simplifyDP(tol, v, j, maxi, mk);
            simplifyDP(tol, v, maxi, k, mk);
        }
        return;
    }
    var n = V.length;
    var sV = [];
    var i, k, m, pv;
    var tol2 = tol * tol;
    var vt = [];
    var mk = [];
    vt[0] = V[0];
    for (i = k = 1, pv = 0; i < n; i++) {
        if (d2(V[i], V[pv]) < tol2) {
            continue;
        }
        vt[k++] = V[i];
        pv = i;
    }
    if (pv < n - 1) {
        vt[k++] = V[n - 1];
    }
    mk[0] = mk[k - 1] = 1;
    simplifyDP(tol, vt, 0, k - 1, mk);
    for (i = m = 0; i < k; i++) {
        if (mk[i]) {
            sV[m++] = vt[i];
        }
    }
    return sV;
}
exports.simplifyLine = simplifyLine;
function simplifyLines(lines, tolerance) {
    var result = [];
    var tol = 2.0;
    if (tolerance) {
        tol = tolerance;
    }
    for (var i = 0; i < lines.length; i++) {
        result.push(simplifyLine(lines[i], tol));
    }
    return result;
}
exports.simplifyLines = simplifyLines;
function linesToStrokes(rawData) {
    var x, y;
    var px = 0, py = 0;
    var dx, dy;
    var pon, poff;
    var stroke = [];
    var i, j;
    var len;
    var p;
    for (i = 0; i < rawData.length; i++) {
        len = rawData[i].length;
        if (len > 1) {
            for (j = 0; j < len; j++) {
                p = rawData[i][j];
                x = p[0];
                y = p[1];
                if (j === len - 1) {
                    poff = 1;
                    pon = 0;
                }
                else {
                    poff = 0;
                    pon = 1;
                }
                dx = x - px;
                dy = y - py;
                px = x;
                py = y;
                stroke.push([dx, dy, pon, poff, 0]);
            }
        }
    }
    stroke.push([0, 0, 0, 0, 1]);
    return stroke.slice(1);
}
exports.linesToStrokes = linesToStrokes;
function lineToStroke(line, lastPoint) {
    var pon, poff;
    var stroke = [];
    var len;
    var p;
    var dx, dy;
    var x, y;
    var px, py;
    var j;
    px = lastPoint[0];
    py = lastPoint[1];
    len = line.length;
    if (len > 1) {
        for (j = 0; j < len; j++) {
            p = line[j];
            x = p[0];
            y = p[1];
            if (j === len - 1) {
                poff = 1;
                pon = 0;
            }
            else {
                poff = 0;
                pon = 1;
            }
            dx = x - px;
            dy = y - py;
            px = x;
            py = y;
            stroke.push([dx, dy, pon, poff, 0]);
        }
    }
    return stroke;
}
exports.lineToStroke = lineToStroke;
//# sourceMappingURL=sketch_support.js.map