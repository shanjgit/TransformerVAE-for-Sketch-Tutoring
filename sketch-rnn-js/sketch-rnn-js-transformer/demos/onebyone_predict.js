/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 /**
 * Author: David Ha <hadavid@google.com>
 *
 * @fileoverview Basic p5.js sketch to show how to use sketch-rnn
 * to finish a fixed incomplete drawings, and loop through multiple
 * endings automatically.
 */

import * as ms from '../src/index';
import * as tf from '@tensorflow/tfjs-core';

const sketch = function(p) {
  // Available SketchRNN models.
  const BASE_URL = 'https://storage.googleapis.com/clouddatawu/';
  const availableModels = ['bird','apple','birdv3','birdv4','birdv5'] //,'fish','dog'] //, 'ant','ambulance','angel','alarm_clock','antyoga','backpack','barn','basket','bear','bee','beeflower','bicycle','book','brain','bridge','bulldozer','bus','butterfly','cactus','calendar','castle','cat','catbus','catpig','chair','couch','crab','crabchair','crabrabbitfacepig','cruise_ship','diving_board','dogbunny','dolphin','duck','elephant','elephantpig','everything','eye','face','fan','fire_hydrant','firetruck','flamingo','flower','floweryoga','frog','frogsofa','garden','hand','hedgeberry','hedgehog','helicopter','kangaroo','key','lantern','lighthouse','lion','lionsheep','lobster','map','mermaid','monapassport','monkey','mosquito','octopus','owl','paintbrush','palm_tree','parrot','passport','peas','penguin','pig','pigsheep','pineapple','pool','postcard','power_outlet','rabbit','rabbitturtle','radio','radioface','rain','rhinoceros','rifle','roller_coaster','sandwich','scorpion','sea_turtle','sheep','skull','snail','snowflake','speedboat','spider','squirrel','steak','stove','strawberry','swan','swing_set','the_mona_lisa','tiger','toothbrush','toothpaste','tractor','trombone','truck','whale','windmill','yoga','yogabicycle'];
  let model;
  let endFlag = false;
  // Model state.
  let modelState; // Store the hidden states of rnn's neurons.
  let temperature = 0.25; // Controls the amount of uncertainty of the model.
  let modelLoaded = false;
  let modelIsActive = false;
  let countStep = 0;
  let stepTh = 300000; // Maximum RNN drawing steps
  let countUserStep = 0;
  let userStepTh = 20; // Maximum drawing steps for user
  let dx, dy; // Offsets of the pen strokes, in pixels.
  let x, y; // Absolute coordinates on the screen of where the pen is.
  let startX, startY;
  let userPen = 0; // above or below the paper
  let previousUserPen = 0;
  let pen = [0,0,0]; // Current pen state, [pen_down, pen_up, pen_end].
  let previousPen = [1, 0, 0]; // Previous pen state.
  let latentZ; // letent variable z extracted by encoder from a image 
  const PEN = {DOWN: 0, UP: 1, END: 2};
  const epsilon = 2.0; // to ignore data from user's pen staying in one spot.

  let userHasEverDrawn = false;
  let allRawLines;
  let currentRawLine = [];
  let strokes;
  /*
   * Main p5 code
   */
  p.setup = function() {
    const containerSize = document.getElementById('sketch').getBoundingClientRect();
    // Initialize the canvas.
    const screenWidth = 256// Math.floor(containerSize.width);
    const screenHeight = 256 //p.windowHeight / 2;
    p.createCanvas(screenWidth, screenHeight);
    p.frameRate(5);

        restart();
        initModel(0);
        initDOMElements();
      };

      /*
      * Human is drawing.
      */
      p.mousePressed = function () {
        if (p.isInBounds()) {
          // First time anything is written.
          if (!userHasEverDrawn) {
            userHasEverDrawn = true;
            x = startX = p.mouseX;
            y = startY = p.mouseY;
            userPen = 1; // down!
          }

          if (endFlag) {
            x = p.mouseX;
            y = p.mouseY;
          }

          modelIsActive = false;
          previousUserPen = userPen;
          p.stroke(p.color(255,0,0));  // User always draws in red.
        }
      }


      p.mouseReleased = function() {
        if (p.isInBounds()) {
          userPen = 0;  // Up!

          const currentRawLineSimplified = model.simplifyLine(currentRawLine);
          let lastX, lastY;

          // If it's an accident...ignore it.
          if (currentRawLineSimplified.length > 1) {
            // Need to keep track of the first point of the last line.
            if (allRawLines.length === 0) {
              lastX = startX;
              lastY = startY;
            } else {
              // The last line.
              const idx = allRawLines.length - 1;
              const lastPoint = allRawLines[idx][allRawLines[idx].length-1];
          lastX = lastPoint[0];
          lastY = lastPoint[1];
        }
        console.log([lastX,lastY]);
        // Encode this line as a stroke, and feed it to the model.
        const stroke = model.lineToStroke(currentRawLineSimplified, [lastX, lastY]);
        allRawLines.push(currentRawLineSimplified);
        strokes = strokes.concat(stroke);

        initRNNStateFromStrokes(strokes);
      }
      currentRawLine = [];
    }
   modelIsActive = true; // uncommnet when using mouseReleased
    previousUserPen = userPen;
  }
  
  function linedash(x1, y1, x2, y2, delta, style = '-') {
  // delta is both the length of a dash, the distance between 2 dots/dashes, and the diameter of a round
  let distance = p.dist(x1,y1,x2,y2);
  let dashNumber = distance/delta;
  let xDelta = (x2-x1)/dashNumber;
  let yDelta = (y2-y1)/dashNumber;

  for (let i = 0; i < dashNumber; i+= 2) {
    let xi1 = i*xDelta + x1;
    let yi1 = i*yDelta + y1;
    let xi2 = (i+1)*xDelta + x1;
    let yi2 = (i+1)*yDelta + y1;

    if (style == '-') { p.line(xi1, yi1, xi2, yi2); }
    else if (style == '.') { p.point(xi1, yi1); }
    else if (style == 'o') { p.ellipse(xi1, yi1, delta/2); }
  }
}


  p.mouseDragged = function () {
    if (!modelIsActive && p.isInBounds()) {
      const dx0 = p.mouseX - x; // Candidate for dx.
      const dy0 = p.mouseY - y; // Candidate for dy.
      if (dx0*dx0+dy0*dy0 > epsilon*epsilon) { // Only if pen is not in same area.
        dx = dx0;
        dy = dy0;
        userPen = 1;
        if (previousUserPen == 1) {
          p.line(x, y, x+dx, y+dy); // draw line connecting prev point to current point.
        }
        x += dx;
        y += dy;
        currentRawLine.push([x, y]);
      }
      previousUserPen = userPen;
    }
  }

 /*
  * Model is drawing.
  */
  p.draw = function() {
     // || !modelIsActive
    if (!modelLoaded || !modelIsActive) {
      return;
    }
    // New state.
    pen = previousPen;
    console.log(previousPen);
    if (pen[PEN.END] === 1 || endFlag){return;} // return if the stroke has ended
    modelState = update([dx, dy, ...pen], modelState, latentZ);
    const NOUT = model.getNMIXTURE();
    const out_o = tf.tidy(()=>{
    const numUnits = model.numUnits;
    const h = tf.tensor3d(modelState.h, [1, 1, numUnits]);
    const out = model.output_model.predict(h);
    return out.reshape([1, 6*NOUT+3]);
    
});
    const pdf = model.getPDF(out_o, temperature);
    out_o.dispose();
    [dx, dy, ...pen] = model.sample(pdf);
    console.log([dx,dy,pen[0],pen[1],pen[2]]);
    // (previousPen[PEN.DOWN] === 1 && pen[PEN.DOWN] === 0)
    // If we finished the previous drawing, start a new one.
    if (pen[PEN.END] === 1 || countStep === stepTh || (previousPen[PEN.DOWN] === 1 && pen[PEN.DOWN] === 0)) {
      if (pen[PEN.END] ===1 && pen[PEN.DOWN]===0){
          if (endFlag){return;}
          endFlag=true;}
      countStep = 0;
      pen[PEN.END] = 1;
      if (!endFlag) {
          x += dx;
          y += dy;}
      currentRawLine.push([x, y]);
      previousPen = pen;
      // Encode this line as a strokes
      const currentRawLineSimplified = model.simplifyLine(currentRawLine);
      var lastX, lastY;
      const idx = allRawLines.length-1;
      const lastPoint = allRawLines[idx][allRawLines[idx].length-1];
      lastX = lastPoint[0];
      lastY = lastPoint[1];
      const stroke = model.lineToStroke(currentRawLineSimplified, [lastX, lastY]);
      allRawLines.push(currentRawLineSimplified);
      strokes = strokes.concat(stroke);
      // initRNNStateFromStrokes(strokes);
      currentRawLine = [];
      drawStrokes(strokes, startX, startY);
      //return;
//    initRNNStateFromStrokes(strokes);
    } else {
      // Only draw on the paper if the pen is still touching the paper.
      if (previousPen[PEN.DOWN] === 1) {
        countStep += 1;
        p.line(x, y, x+dx, y+dy); // Draw line connecting prev point to current point.
       }
      // Update.
      x += dx;
      y += dy;
      previousPen = pen;
      currentRawLine.push([x, y]);
    }
  };

  p.isInBounds = function () {
    return p.mouseX >= 0 && p.mouseY >= 0;
  }


   /*
   * Helpers.
   */
  function restart() {
    endFlag=false;
    p.background(255, 255, 255, 255);
    p.strokeWeight(3.0);

    // Start drawing in the middle-ish of the screen
    startX = x = p.width / 2.0;
    startY = y = p.height / 3.0;

    // Reset the user drawing state.
    userPen = 1;
    previousUserPen = 0;
    userHasEverDrawn = false;
    allRawLines = [];
    currentRawLine = [];
    strokes = [];

    // Reset the model drawing state.
    modelIsActive = false;
    previousPen = [0, 1, 0];
  };

  function initRNNStateFromStrokes(strokes) {
    // Initialize the RNN with these strokes.
    encodeStrokes(strokes);
    // Draw them.
    p.background(255, 255, 255, 255);
    drawStrokes(strokes, startX, startY);
  }


/**
* High pass filtering the input image 
* @param {imgIn}: [N, H, W, 1]
* @return {imgOut}; [N, H, W, 1]
*/
function highPassFiltering(imgIn){
    return tf.tidy(() => {
      const shape = [3,3];
      const type = 'float32';
      const strides = [1,1];
      const pad = 'same';
      const filter = tf.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], shape, type);
      const filterHp = tf.expandDims(tf.expandDims(filter, -1), -1);
      const imgOut = tf.conv2d(imgIn, filterHp, strides, pad);
      return imgOut;

    });
}
 
 function initModel(index) {
    modelLoaded = false;
    if (model) {
      model.dispose();
    }
    model = new ms.SketchRNN(`${BASE_URL}${availableModels[index]}`);

    Promise.all([model.initialize()]).then(function() {
      modelLoaded = true;
      console.log('SketchRNN model loaded.');

      // Initialize the scale factor for the model. Bigger -> large outputs
      model.setPixelFactor(2);

      var imgTensor = tf.tensor(model.imgArray);
      imgTensor = highPassFiltering(imgTensor)
      var out = model.encoder.predict(tf.expandDims(imgTensor,0));
     
      var sigma_exp = tf.exp(out[1].div(tf.scalar(2.0)));
      latentZ = tf.add(out[0], sigma_exp.mul(tf.randomNormal(sigma_exp.shape, 0.0, 1.0)));
      if (strokes.length > 0) {
        initRNNStateFromStrokes(strokes);
      }
    });
  };

  function initRNNEvent() {
    initRNNStateFromStrokes(strokes);
   }
 


/**
   * Updates the RNN, returns the next state.
   *
   * @param stroke [dx, dy, penDown, penUp, penEnd].
   * @param state previous LSTMState.
   *
   * @returns next LSTMState.
   */
function  update(stroke, state, z) {
    if (typeof state =='undefined'){state = model.zeroState();}
    const out = tf.tidy(() => {
      const numUnits = model.numUnits;
      const s = model.scaleFactor;
      const normStroke =
        [stroke[0]/s, stroke[1]/s, stroke[2], stroke[3], stroke[4]];
      const x = tf.tensor3d(normStroke, [1, 1, 5]);
      const c = tf.tensor2d(state.c, [1, numUnits]);
      const h = tf.tensor2d(state.h, [1, numUnits]);
      const newState = model.decoder.predict([x, h, c, z]);
      return newState[1].concat(newState[2], 1);
    });
    const newCH = out.dataSync();
    out.dispose();
    const newH = newCH.slice(0, model.numUnits);
    const newC = newCH.slice(model.numUnits, model.numUnits * 2);
    const finalState = model.zeroState();
    finalState.c = Array.from(newC);
    finalState.h = Array.from(newH);
    return finalState;
  }
   

  /**
   * Updates the RNN on a series of Strokes, returns the next state.
   *
   * @param strokes list of [dx, dy, penDown, penUp, penEnd].
   * @param state previous LSTMState.
   * @param steps (Optional) number of steps of the stroke to update
   * (default is length of strokes list)
   * 
   *
   * @returns the final LSTMState.
   */
 function updateStrokes(strokes, state, steps, z) {
    const out = tf.tidy(() => {
      const numUnits = model.numUnits;
      const s = model.scaleFactor;
      let normStroke;
      let x;
      let c;
      let h;
      let newState;
      let numSteps = strokes.length;
      if (steps) {
        numSteps = steps;
      }
      c = tf.tensor2d(state.c, [1, numUnits]);
      h = tf.tensor2d(state.h, [1, numUnits]);
      for (let i=0;i<numSteps;i++) {
      normStroke = [strokes[i][0]/s,
                      strokes[i][1]/s,
                      strokes[i][2],
                      strokes[i][3],
                      strokes[i][4]];
        x = tf.tensor3d(normStroke, [1, 1, 5]);
        newState = model.decoder.predict([x, h, c, z])
        c = newState[2];
        h = newState[1];
      }
      return h.concat(c, 1);
    });
    const newHC = out.dataSync();
    out.dispose();
    const newH = newHC.slice(0, model.numUnits);
    const newC = newHC.slice(model.numUnits, model.numUnits * 2);
    const finalState = model.zeroState();
    finalState.h = Array.from(newH);
    finalState.c = Array.from(newC);
    return finalState;
  }

  function splitAndMakeLSTMState(state) {
      const newHC = out.dataSync();
      const newH = newHC.slice(0, model.numUnits);
      const newC = newHC.slice(model.numUnits, model.numUnits * 2);
      const finalState = model.zeroState();
      finalState.h = Array.from(newH);
      finalState.c = Array.from(newC);
      return finalState;
}

  
  function encodeStrokes(sequence) {
    if (sequence.length <= 5) {
      return;
    }

    // Encode the strokes in the model.
    let newState = model.zeroState();
    const newS = model.z_to_init.predict(latentZ);
    const newHC = newS.dataSync();
    const newH = newHC.slice(0, model.numUnits);
    const newC = newHC.slice(model.numUnits, model.numUnits * 2);
    newState.h = Array.from(newH);
    newState.c = Array.from(newC);
    newState = update(model.zeroInput(), newState, latentZ);
    newState = updateStrokes(sequence, newState, sequence.length-1, latentZ);

    // Reset the actual model we're using to this one that has the encoded strokes.
    modelState = model.copyState(newState);

    // Reset the state.
    const idx = allRawLines.length - 1;
    const lastPoint = allRawLines[idx][allRawLines[idx].length-1];
    x = lastPoint[0];
    y = lastPoint[1];

    const s = sequence[sequence.length-1];
    dx = s[0];
    dy = s[1];
    previousPen = [s[2], s[3], s[4]];

    modelIsActive = true; // true in normal mode!!!
  }

  // This is very similar to the p.draw() loop, but instead of
  // sampling from the model, it uses the given set of strokes.
  function drawStrokes(strokes, startX, startY) {
    p.stroke(p.color(255,0,0));

    let x = startX;
    let y = startY;
    let dx, dy;
    let pen = [0,0,0];
    let previousPen = [1,0,0];
    for( let i = 0; i < strokes.length; i++) {
      [dx, dy, ...pen] = strokes[i];

      if (previousPen[PEN.END] === 1) { // End of drawing.
          console.log('end of pen');
          break;
      }

      // Only draw on the paper if the pen is still touching the paper.
      if (previousPen[PEN.DOWN] === 1) {
        p.line(x, y, x+dx, y+dy);
      }
      x += dx;
      y += dy;
      previousPen = pen;
    }

    // Draw in a random colour after the predefined strokes.
    p.stroke(p.color(p.random(64, 224), p.random(64, 224), p.random(64, 224)));
  };

 function initDOMElements() {
    // Setup the DOM bits.
    // textTemperature.textContent = inputTemperature.value = temperature;

    // Listeners
    selectModels.innerHTML = availableModels.map(m => `<option>${m}</option>`).join('');
    selectModels.addEventListener('change', () =>{ 
        restart();
        initModel(selectModels.selectedIndex);});
    // inputTemperature.addEventListener('change', () => {
    //  temperature = parseFloat(inputTemperature.value);
    //  textTemperature.textContent = temperature;
    //});
    btnClear.addEventListener('click', restart)
//    btnRandom.addEventListener('click', () => {
//      selectModels.selectedIndex = Math.floor(Math.random() * availableModels.length);
//      initModel(selectModels.selectedIndex);
//    });
  }
};

new p5(sketch, 'sketch');
