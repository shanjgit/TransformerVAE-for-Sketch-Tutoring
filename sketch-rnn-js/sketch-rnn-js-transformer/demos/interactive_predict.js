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
import * as tfl from '@tensorflow/tfjs-layers';

const sketch = function(p) {
	// Available SketchRNN models.
	const BASE_URL = 'https://storage.googleapis.com/clouddatawu/';
	const availableModels = ['cat_transformer', 'cup_transformer', 'sun_transformer', 'spoon_transformer', 'bird_transformer_new', 'basketball_transformer', 'apple_transformer'] //'bird_transformer_new',,'fish','dog'] //, 'ant','ambulance','angel','alarm_clock','antyoga','backpack','barn','basket','bear','bee','beeflower','bicycle','book','brain','bridge','bulldozer','bus','butterfly','cactus','calendar','castle','cat','catbus','catpig','chair','couch','crab','crabchair','crabrabbitfacepig','cruise_ship','diving_board','dogbunny','dolphin','duck','elephant','elephantpig','everything','eye','face','fan','fire_hydrant','firetruck','flamingo','flower','floweryoga','frog','frogsofa','garden','hand','hedgeberry','hedgehog','helicopter','kangaroo','key','lantern','lighthouse','lion','lionsheep','lobster','map','mermaid','monapassport','monkey','mosquito','octopus','owl','paintbrush','palm_tree','parrot','passport','peas','penguin','pig','pigsheep','pineapple','pool','postcard','power_outlet','rabbit','rabbitturtle','radio','radioface','rain','rhinoceros','rifle','roller_coaster','sandwich','scorpion','sea_turtle','sheep','skull','snail','snowflake','speedboat','spider','squirrel','steak','stove','strawberry','swan','swing_set','the_mona_lisa','tiger','toothbrush','toothpaste','tractor','trombone','truck','whale','windmill','yoga','yogabicycle'];
	let model;

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
	let draw_strokes;
	

	let input_to_embed_dense;
	let output_model;
	let wq_dense = [0, 0, 0 ,0];
	let wk_dense = [0, 0, 0 ,0];
	let wv_dense = [0, 0, 0 ,0];
	let concat_attention_to_decoder_output = [0, 0, 0 ,0];

	
	/*
	 * Main p5 code
	 */
	p.setup = function() {
		const containerSize = document.getElementById('sketch').getBoundingClientRect();
		// Initialize the canvas.
		const screenWidth = 256// Math.floor(containerSize.width);
		const screenHeight = 256 //p.windowHeight / 2;
		p.createCanvas(screenWidth, screenHeight);
		p.frameRate(60);

		restart();
		initModel(0);
		initDOMElements();
	};

	/*
	* Human is drawing.
	*/
	p.mousePressed = function () {
		console.log('mousePressed')
		if (p.isInBounds()) {
			// First time anything is written.
				if (!userHasEverDrawn) {
						userHasEverDrawn = true;
				x = startX = p.mouseX;
				y = startY = p.mouseY;
				userPen = 1; // down!
			}

			modelIsActive = false;
			previousUserPen = userPen;
			p.stroke(p.color(255,0,0));  // User always draws in red.
		}
	}
//  p.mouseReleased = function(){userPen = 0;  previousUserPen = userPen;}


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

				// Encode this line as a stroke, and feed it to the model.
				const stroke = model.lineToStroke(currentRawLineSimplified, [lastX, lastY]);
				allRawLines.push(currentRawLineSimplified);
				strokes = strokes.concat(stroke);
				draw_strokes = strokes;
				initRNNStateFromStrokes(strokes);
			}
			currentRawLine = [];
		}
		//modelIsActive = true; // uncommnet when using mouseReleased
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
			//countUserStep += 1;
			//if (countUserStep >= userStepTh){countUserStep = 0; p.giveTutorial()}
		}
	}

 /*
	* Model is drawing.
	*/
	p.draw = function() {
		 // || !modelIsActive
		//console.log(modelLoaded, modelIsActive)
		if (!modelLoaded || !modelIsActive) {
			return;
		}
		//console.log('drawing' , draw_strokes)
		// New state.
		pen = previousPen;
		

		if (pen[PEN.END] === 1){return;} // return if the stroke has ended
		//modelState = update([dx, dy, ...pen], modelState, latentZ);
		const NOUT = model.getNMIXTURE();


		const out_o = tf.tidy(()=>{

		const s = model.scaleFactor;
		

		//console.log('draw_strokes', draw_strokes)
		let numSteps = draw_strokes.length;
		//console.log('numSteps', numSteps)

		var input = [];
		input = input.concat(model.zeroInput());
		//console.log('input', input);


		let normStroke;
		for (let i=0;i<numSteps;i++) {
			normStroke = [draw_strokes[i][0]/s, draw_strokes[i][1]/s, draw_strokes[i][2], draw_strokes[i][3], draw_strokes[i][4]];
			input = input.concat(normStroke);
		}
		//console.log('input', input);
		//console.log('normStroke', normStroke);
		input = tf.tensor3d(input, [1, numSteps+1, 5]);
		//console.log('input_tensor', input);


		/*
		var input = new Array(numSteps+1);
		input[0] = model.zeroInput();
		console.log('input', input);


		let normStroke;
		for (let i=0;i<numSteps;i++) {
			normStroke = [strokes[i][0]/s, strokes[i][1]/s, strokes[i][2], strokes[i][3], strokes[i][4]];
			input[i+1] = normStroke;
		}
		console.log('input', input);
		//console.log('normStroke', normStroke);
		input = tf.tensor3d(input, [1, numSteps+1, 5]);
		console.log('input_tensor', input);
		*/



		//let tmp = z.concat(normStroke)
		//const tmp = tf.tensor3d(z, [1, 1, 64]);
		//x = tf.expandDims(latentZ)
		//console.log('x', x)
		let z = latentZ;
		for (let i=0;i<numSteps;i++)
			z = z.concat(latentZ);
		//console.log('z', z)
		z = tf.expandDims(z);
		let full_input = input.concat(z,2);
		//let full_input = tf.tensor3d(tmp, [1,1,69]);

		//const values1 = full_input.dataSync();
		//const arr1 = Array.from(values1);
		//console.log(values1);
		//console.log(arr1);
	 


		//full_input = tf.tensor3d(arr1, [1,1,69])

		//console.log('full_input', full_input)
		const embed_output = model.input_to_embed_dense.predict(full_input)
		//console.log('embed_output', embed_output)

	 
		

		//const numUnits = model.numUnits;
		//const h = tf.tensor3d(modelState.h, [1, 1, numUnits]);
		//console.log(strokes)
		let seq_len = full_input.shape[1]
		let look_ahead_mask = lookAheadMask(seq_len);
		let pos_encoding = positionalEncoding(112, 256);//tf.tensor(get_pos_encode());
		let embed_output_new_1 = embed_output.mul(tf.sqrt(tf.cast(256,'float32')));

		

		let embed_output_new = embed_output_new_1.add(pos_encoding.slice([0, 0, 0], [-1, seq_len, -1]))

		

		let decoder_output = embed_output_new
		for(let i = 0 ; i < 4 ; i++){
			console.log('decoder_output', decoder_output)
			let batch_size = decoder_output.shape[0]
			let q =  model.wq_dense[i].predict(decoder_output)
			let k =  model.wk_dense[i].predict(decoder_output)
			let v =  model.wv_dense[i].predict(decoder_output)

			let qs = split_heads(q, batch_size, 256, 8)
			let ks = split_heads(k, batch_size, 256, 8) 
			let vs = split_heads(v, batch_size, 256, 8)


			let tmp = scaled_dot_product_attention(qs, ks, vs, look_ahead_mask)
			let scaled_attention = tmp[0]
			let attn_weights_block = tmp[1]
			scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
			scaled_attention = tf.reshape(scaled_attention, [batch_size, -1, 256])

			decoder_output = model.concat_attention_to_decoder_output[i].predict([scaled_attention, decoder_output])
			
		}
		//console.log('decoder_output', decoder_output)

		const out = model.output_model.predict(decoder_output);
		//console.log(out);
		const tmp_out = out.slice([0,numSteps]);
		//console.log(tmp_out);
		return tmp_out.reshape([1, 6*NOUT+3]);
		
});
		


		const pdf = model.getPDF(out_o, temperature);
		console.log('success')
		out_o.dispose();
		[dx, dy, ...pen] = model.sample(pdf);
		let new_stroke = [dx, dy, ...pen];  
		console.log('new_stroke', new_stroke)
		draw_strokes = draw_strokes.concat([new_stroke]);
		console.log('new_draw_strokes', draw_strokes)
		console.log('strokes', strokes)
		console.log('sampled');
		console.log([dx, dy]);
		// (previousPen[PEN.DOWN] === 1 && pen[PEN.DOWN] === 0)
		// If we finished the previous drawing, start a new one.
		// PEN = {DOWN: 0, UP: 1, END: 2};
		
		if (pen[PEN.END] === 1 || (previousPen[PEN.DOWN] === 1 && pen[PEN.DOWN] === 0)){// || (previousPen[PEN.DOWN] === 1 && pen[PEN.DOWN] === 0)) 
			countStep = 0;
			pen[PEN.END] = 1;
			previousPen = pen;
//    return;
//    initRNNStateFromStrokes(strokes);
		} else {
		
			// Only draw on the paper if the pen is still touching the paper.
			if (previousPen[PEN.DOWN] === 1) {
				countStep += 1;
				// p.line(x, y, x+dx, y+dy); // Draw line connecting prev point to current point.
				linedash(x, y, x+dx, y+dy, 2,'.')
			 }
			// Update.
			x += dx;
			y += dy;
			previousPen = pen;
		}
	};

	p.isInBounds = function () {
		return p.mouseX >= 0 && p.mouseY >= 0;
	}
	 /*
	 * Helpers.
	 */
	function restart() {
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
function download(content, fileName, contentType) {
		var a = document.createElement("a");
		var file = new Blob([content], {type: contentType});
		a.href = URL.createObjectURL(file);
		a.download = fileName;
		a.click();
}
 
 function initModel(index) {
		modelLoaded = false;
		if (model) {
			model.dispose();
		}
		console.log('start init?');
		console.log(`${BASE_URL}${availableModels[index]}`);
		model = new ms.SketchRNN(`${BASE_URL}${availableModels[index]}`);

		Promise.all([model.initialize()]).then(function() {
			modelLoaded = true;
			console.log('SketchRNN model loaded.');
			/*
			var dense = model.input_to_embed_dense.getLayer('dense_2');
			var w = dense.getWeights();
			input_to_embed_dense = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,69]})]
			});
			input_to_embed_dense.layers[0].setWeights(w);


			dense = model.output_model.getLayer('output');
			w = dense.getWeights();
			output_model = tfl.sequential({
					layers: [tfl.layers.dense({units: 123, inputShape: [null,256]})]
			});
			output_model.layers[0].setWeights(w);

			//wq wk wv
			dense = model.wq_dense[0].getLayer('dense_27');
			w = dense.getWeights();
			wq_dense[0] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wq_dense[0].layers[0].setWeights(w);

			dense = model.wq_dense[1].getLayer('dense_28');
			w = dense.getWeights();
			wq_dense[1] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wq_dense[1].layers[0].setWeights(w);

			dense = model.wq_dense[2].getLayer('dense_29');
			w = dense.getWeights();
			wq_dense[2] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wq_dense[2].layers[0].setWeights(w);

			dense = model.wq_dense[3].getLayer('dense_30');
			w = dense.getWeights();
			wq_dense[3] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wq_dense[3].layers[0].setWeights(w);


			dense = model.wk_dense[0].getLayer('dense_31');
			w = dense.getWeights();
			wk_dense[0] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wk_dense[0].layers[0].setWeights(w);

			dense = model.wk_dense[1].getLayer('dense_32');
			w = dense.getWeights();
			wk_dense[1] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wk_dense[1].layers[0].setWeights(w);

			dense = model.wk_dense[2].getLayer('dense_33');
			w = dense.getWeights();
			wk_dense[2] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wk_dense[2].layers[0].setWeights(w);

			dense = model.wk_dense[3].getLayer('dense_34');
			w = dense.getWeights();
			wk_dense[3] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wk_dense[3].layers[0].setWeights(w);
			


			dense = model.wv_dense[0].getLayer('dense_35');
			w = dense.getWeights();
			wv_dense[0] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wv_dense[0].layers[0].setWeights(w);

			dense = model.wv_dense[1].getLayer('dense_36');
			w = dense.getWeights();
			wv_dense[1] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wv_dense[1].layers[0].setWeights(w);

			dense = model.wv_dense[2].getLayer('dense_37');
			w = dense.getWeights();
			wv_dense[2] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wv_dense[2].layers[0].setWeights(w);

			dense = model.wv_dense[3].getLayer('dense_38');
			w = dense.getWeights();
			wv_dense[3] = tfl.sequential({
					layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			});
			wv_dense[3].layers[0].setWeights(w);



			//concat_attention_to_decoder_output
			
			//concat_attention_to_decoder_output[0] = tfl.sequential({
			//    layers: [tfl.layers.dense({units: 256, inputShape: [null,256]})]
			//});
			var input_1 = tfl.input({shape: [null, 256]});
			var input_2 = tfl.input({shape: [null, 256]});
			var dense_layer_1 = tfl.layers.dense({units: 256, inputShape: [null,256]});
			var dense_layer_2 = tfl.layers.dense({units: 1024, inputShape: [null,256], activation: 'relu'});
			var dense_layer_3 = tfl.layers.dense({units: 256, inputShape: [null,1024]});
			var layer_norm_1 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var layer_norm_2 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var add_1 = tfl.layers.add();
			var add_2 = tfl.layers.add();
		 
			var tmp = dense_layer_1.apply(input_1);
			tmp = add_1.apply([tmp, input_2]);
			var tmp_out = layer_norm_1.apply(tmp);

			tmp = dense_layer_2.apply(tmp_out);
			tmp = dense_layer_3.apply(tmp);
			tmp = add_2.apply([tmp, tmp_out]);
			var output = layer_norm_2.apply(tmp);
			

			concat_attention_to_decoder_output[0] = tfl.model({inputs: [input_1, input_2], outputs: output});
			console.log(concat_attention_to_decoder_output[0].layers)

			var layer = model.concat_attention_to_decoder_output[0].getLayer('dense_39');
			w = layer.getWeights();
			concat_attention_to_decoder_output[0].layers[1].setWeights(w);


			layer = model.concat_attention_to_decoder_output[0].getLayer('layer_normalization_8');
			w = layer.getWeights();
			concat_attention_to_decoder_output[0].layers[4].setWeights(w);


			layer = model.concat_attention_to_decoder_output[0].getLayer('dense_43');
			w = layer.getWeights();
			concat_attention_to_decoder_output[0].layers[5].setWeights(w);


			layer = model.concat_attention_to_decoder_output[0].getLayer('dense_47');
			w = layer.getWeights();
			concat_attention_to_decoder_output[0].layers[6].setWeights(w);


			layer = model.concat_attention_to_decoder_output[0].getLayer('layer_normalization_12');
			w = layer.getWeights();
			concat_attention_to_decoder_output[0].layers[8].setWeights(w);



			var input_1 = tfl.input({shape: [null, 256]});
			var input_2 = tfl.input({shape: [null, 256]});
			var dense_layer_1 = tfl.layers.dense({units: 256, inputShape: [null,256]});
			var dense_layer_2 = tfl.layers.dense({units: 1024, inputShape: [null,256], activation: 'relu'});
			var dense_layer_3 = tfl.layers.dense({units: 256, inputShape: [null,1024]});
			var layer_norm_1 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var layer_norm_2 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var add_1 = tfl.layers.add();
			var add_2 = tfl.layers.add();
		 
			var tmp = dense_layer_1.apply(input_1);
			tmp = add_1.apply([tmp, input_2]);
			var tmp_out = layer_norm_1.apply(tmp);

			tmp = dense_layer_2.apply(tmp_out);
			tmp = dense_layer_3.apply(tmp);
			tmp = add_2.apply([tmp, tmp_out]);
			var output = layer_norm_2.apply(tmp);
			
			concat_attention_to_decoder_output[1] = tfl.model({inputs: [input_1, input_2], outputs: output});

			var layer = model.concat_attention_to_decoder_output[1].getLayer('dense_40');
			w = layer.getWeights();
			concat_attention_to_decoder_output[1].layers[1].setWeights(w);


			layer = model.concat_attention_to_decoder_output[1].getLayer('layer_normalization_9');
			w = layer.getWeights();
			concat_attention_to_decoder_output[1].layers[4].setWeights(w);


			layer = model.concat_attention_to_decoder_output[1].getLayer('dense_44');
			w = layer.getWeights();
			concat_attention_to_decoder_output[1].layers[5].setWeights(w);


			layer = model.concat_attention_to_decoder_output[1].getLayer('dense_48');
			w = layer.getWeights();
			concat_attention_to_decoder_output[1].layers[6].setWeights(w);


			layer = model.concat_attention_to_decoder_output[1].getLayer('layer_normalization_13');
			w = layer.getWeights();
			concat_attention_to_decoder_output[1].layers[8].setWeights(w);



			var input_1 = tfl.input({shape: [null, 256]});
			var input_2 = tfl.input({shape: [null, 256]});
			var dense_layer_1 = tfl.layers.dense({units: 256, inputShape: [null,256]});
			var dense_layer_2 = tfl.layers.dense({units: 1024, inputShape: [null,256], activation: 'relu'});
			var dense_layer_3 = tfl.layers.dense({units: 256, inputShape: [null,1024]});
			var layer_norm_1 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var layer_norm_2 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var add_1 = tfl.layers.add();
			var add_2 = tfl.layers.add();
		 
			var tmp = dense_layer_1.apply(input_1);
			tmp = add_1.apply([tmp, input_2]);
			var tmp_out = layer_norm_1.apply(tmp);

			tmp = dense_layer_2.apply(tmp_out);
			tmp = dense_layer_3.apply(tmp);
			tmp = add_2.apply([tmp, tmp_out]);
			var output = layer_norm_2.apply(tmp);
			



			concat_attention_to_decoder_output[2] = tfl.model({inputs: [input_1, input_2], outputs: output});

			var layer = model.concat_attention_to_decoder_output[2].getLayer('dense_41');
			w = layer.getWeights();
			concat_attention_to_decoder_output[2].layers[1].setWeights(w);


			layer = model.concat_attention_to_decoder_output[2].getLayer('layer_normalization_10');
			w = layer.getWeights();
			concat_attention_to_decoder_output[2].layers[4].setWeights(w);


			layer = model.concat_attention_to_decoder_output[2].getLayer('dense_45');
			w = layer.getWeights();
			concat_attention_to_decoder_output[2].layers[5].setWeights(w);


			layer = model.concat_attention_to_decoder_output[2].getLayer('dense_49');
			w = layer.getWeights();
			concat_attention_to_decoder_output[2].layers[6].setWeights(w);


			layer = model.concat_attention_to_decoder_output[2].getLayer('layer_normalization_14');
			w = layer.getWeights();
			concat_attention_to_decoder_output[2].layers[8].setWeights(w);




			var input_1 = tfl.input({shape: [null, 256]});
			var input_2 = tfl.input({shape: [null, 256]});
			var dense_layer_1 = tfl.layers.dense({units: 256, inputShape: [null,256]});
			var dense_layer_2 = tfl.layers.dense({units: 1024, inputShape: [null,256], activation: 'relu'});
			var dense_layer_3 = tfl.layers.dense({units: 256, inputShape: [null,1024]});
			var layer_norm_1 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var layer_norm_2 = tfl.layers.layerNormalization({epsilon: 1e-6});
			var add_1 = tfl.layers.add();
			var add_2 = tfl.layers.add();
		 
			var tmp = dense_layer_1.apply(input_1);
			tmp = add_1.apply([tmp, input_2]);
			var tmp_out = layer_norm_1.apply(tmp);

			tmp = dense_layer_2.apply(tmp_out);
			tmp = dense_layer_3.apply(tmp);
			tmp = add_2.apply([tmp, tmp_out]);
			var output = layer_norm_2.apply(tmp);
			


			concat_attention_to_decoder_output[3] = tfl.model({inputs: [input_1, input_2], outputs: output});

			var layer = model.concat_attention_to_decoder_output[3].getLayer('dense_42');
			w = layer.getWeights();
			concat_attention_to_decoder_output[3].layers[1].setWeights(w);


			layer = model.concat_attention_to_decoder_output[3].getLayer('layer_normalization_11');
			w = layer.getWeights();
			concat_attention_to_decoder_output[3].layers[4].setWeights(w);


			layer = model.concat_attention_to_decoder_output[3].getLayer('dense_46');
			w = layer.getWeights();
			concat_attention_to_decoder_output[3].layers[5].setWeights(w);


			layer = model.concat_attention_to_decoder_output[3].getLayer('dense_50');
			w = layer.getWeights();
			concat_attention_to_decoder_output[3].layers[6].setWeights(w);


			layer = model.concat_attention_to_decoder_output[3].getLayer('layer_normalization_15');
			w = layer.getWeights();
			concat_attention_to_decoder_output[3].layers[8].setWeights(w);


			var try_input_1 = tf.ones([1,10,256]);
			var try_input_11 = tf.ones([1,10,256]);
			var try_input_2 = tf.ones([1,10,256]);
			var try_input_21 = tf.ones([1,10,256]);

			var try_output_1 = concat_attention_to_decoder_output[0].predict([try_input_1, try_input_11]);
			var try_output_2 = model.concat_attention_to_decoder_output[0].predict([try_input_2, try_input_21]);
			// var try_output_1 = wk_dense[1].predict(try_input_1);
			// var try_output_2 = model.wk_dense[1].predict(try_input_2);
			console.log(try_output_1.print());
			console.log(try_output_2.print());
			if(try_output_1 == try_output_2){
				console.log('the same !!!!')
			}
			console.log('transfer');
			*/
			/*
			// test
			$.getJSON('https://storage.googleapis.com/clouddatawu/bird_transformer_new/test_func.json', function(data) {
				console.log(data['pos_encoding']);
			});

			var json_data = {}

			var pos_encoding = positionalEncoding(112, 256);
			pos_encoding = pos_encoding.arraySync();
			console.log(pos_encoding);
			json_data['pos_encoding'] = pos_encoding;

			var look_ahead_mask = lookAheadMask(100);
			console.log(look_ahead_mask);
			look_ahead_mask = look_ahead_mask.arraySync();
			console.log(look_ahead_mask);
			json_data['mask'] = look_ahead_mask;

			var x = tf.ones([1,100,256]);
			var sh = split_heads(x, 1, 256, 8);
			sh = sh.arraySync();
			console.log(sh);
			json_data['split_heads'] = sh;

			var q =  tf.ones([1,8,100,32]);
			var k =  tf.ones([1,8,100,32]);
			var v =  tf.ones([1,8,100,32]);
			var mask = lookAheadMask(100);
			var sdpa = scaled_dot_product_attention(q,k,v,mask);
			var sdpa_0 = sdpa[0].arraySync();
			console.log(sdpa_0);
			json_data['sdpa_0'] = sdpa_0;

			var jsonData = JSON.stringify(json_data);
			download(jsonData, 'json.txt', 'text/plain');
			*/


			// Initialize the scale factor for the model. Bigger -> large outputs
			model.setPixelFactor(4); // 4 -> 2

			var imgTensor = tf.tensor(model.imgArray);
			imgTensor = highPassFiltering(imgTensor)
			var out = model.encoder.predict(tf.expandDims(imgTensor,0));
		 
			var sigma_exp = tf.exp(out[1].div(tf.scalar(2.0)));
			latentZ = tf.add(out[0], sigma_exp.mul(tf.randomNormal(sigma_exp.shape, 0.0, 1.0)));
			if (strokes.length > 0) {
				console.log('strokes length > 0');
				initRNNStateFromStrokes(strokes);
			}
		});
	};






	function encodeStrokes(sequence) {
		//if (sequence.length <= 5) {
		//	return;
		//}

		// Encode the strokes in the model.
		//let newState = model.zeroState();
		//console.log(model.zeroInput().length)
		//const newS = tf.tensor(model.zeroInput());
		//for(let i = 0 ; i < model.zeroInput().length; i++){
		//  newS[i] = latentZ;
		//}
		//console.log(newS)
		//const newS = model.z_to_init.predict(latentZ);
		//const newHC = newS.dataSync();
		//const newH = newHC.slice(0, model.numUnits);
		//const newC = newHC.slice(model.numUnits, model.numUnits * 2);
		//newState.h = Array.from(newH);
		//newState.c = Array.from(newC);
		//newState = update(model.zeroInput(), newState, latentZ);
		//newState = updateStrokes(sequence, newState, sequence.length-1, latentZ);

		// Reset the actual model we're using to this one that has the encoded strokes.
		//modelState = model.copyState(newState);

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
		console.log('drawstrokes');

		let x = startX;
		let y = startY;
		let dx, dy;
		let pen = [0,0,0];
		let previousPen = [1,0,0];
		for( let i = 0; i < strokes.length; i++) {
			[dx, dy, ...pen] = strokes[i];

			if (previousPen[PEN.END] === 1) { // End of drawing.
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
		selectModels.addEventListener('change', () => {
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
function split_heads(x, batch_size, dec_size, num_heads){
	var depth = Math.floor(dec_size/num_heads);
	var output = x.reshape([batch_size, -1, num_heads, depth]);

	return tf.transpose(output, [0, 2, 1, 3]);
}

function lookAheadMask(seqLen){
	const inputOnes = tf.ones([seqLen, seqLen]);
	const mask = tf.ones([seqLen, seqLen]).sub(tf.linalg.bandPart(inputOnes, -1, 0));
	return mask;
}

function scaled_dot_product_attention(q, k, v, mask){
	var matmulQk = q.matMul(k, false, true);
	
	// scale matmul_qk
	var kShape = k.shape;
	var dk = tf.cast(kShape[kShape.length-1], 'float32');
	var scaledAttentionLogits = matmulQk.div(tf.sqrt(dk));

	// add the mask to the scaled tensor.
	if (mask != null){
		scaledAttentionLogits = scaledAttentionLogits.add(mask.mul(tf.scalar(-1e9))); }
		 
	// softmax is normalized on the last axis (seq_len_k) so that the scores
	// add up to 1.
	var attentionWeights = scaledAttentionLogits.softmax();  // (..., seq_len_q, seq_len_k)
	// console.log(attentionWeights.shape); // [3,5,4] 
	

	var output = tf.matMul(attentionWeights, v); // (..., seq_len_q, depth_v)

	return [output, attentionWeights];

}

function getAngles(pos, i, dModel){
	var tmp = tf.mul(tf.scalar(2), i.floorDiv(2));
	var tmp2 = tf.div(tmp, tf.scalar(dModel,'float32'))
	var angle_rates = tf.div(1,tf.pow(tf.scalar(10000), tmp2));

	return tf.mul(pos, angle_rates);

}


/** Calculate the positioinal encoding 
* 
* Args
* position, dModel: int32
* return: 2D tf.Tensor radial angle, shape = [1, position, dMomel]
*/
function positionalEncoding(position, dModel){
	var angle_rads = getAngles(tf.expandDims(tf.range(0,position),1),
								tf.expandDims(tf.range(0,dModel),0),
								dModel);
	var p = angle_rads.transpose();
	const cond = tf.tensor1d(Array.from({length: p.shape[0]}, 
																				 (_, i) => i%2 === 1), 'bool');

	// apply sin to even indices in the array; 2i
	// angle_rads[:, 0::2] = tf.sin(angle_rads[:, 0::2]);

	// apply cos to odd indices in the array; 2i+1
	const newp = p.cos().where(cond, p.sin());

	var pos_encoding = tf.expandDims(newp.transpose(),0);
	return tf.cast(pos_encoding, 'float32');
}
new p5(sketch, 'sketch');
