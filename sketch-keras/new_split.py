from new_seq2seqVAE import *
from utils import *
import json
import sys
import argparse
import os
import copy
from new_seq2seqVAE_train import *
#from transformer_decoder import *
from render_svg2bitmap import *
import numpy as np
import cv2
from PIL import Image
import svgwrite 
import random
random.seed(42)

def draw_strokes(data, padding = 50, factor=0.2, svg_filename = 'tmp/sketch_rnn/svg/sample.svg'):
	if not os.path.exists(os.path.dirname(svg_filename)):
		os.makedirs(os.path.dirname(svg_filename))
	min_x, max_x, min_y, max_y = get_bounds(data, factor)
	dims = (padding + max_x - min_x, padding + max_y - min_y)
	dwg = svgwrite.Drawing(svg_filename, size=dims)
	dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
	lift_pen = 1
	abs_x = padding / 2 - min_x 
	abs_y = padding / 2 - min_y
	p = "M%s,%s " % (abs_x, abs_y)
	command = "m"
	for i in range(len(data)):
		if (lift_pen == 1):
			command = "m"
		elif (command != "l"):
			command = "l"
		else:
			command = ""
		x = float(data[i,0])/factor
		y = float(data[i,1])/factor
		lift_pen = data[i, 2]
		p += command+str(x)+","+str(y)+" "
	the_color = "black"
	stroke_width = 1
	dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
	dwg.save()
	#display(SVG(dwg.tostring()))
	return dims, dwg.tostring()

def encode(image):
	
	return seq2seq.sample_models['encoder_model'].predict(image)	

# Function for decoding a latent space factor into a sketch
def decode(z_input=None, draw_mode=False, temperature=0.1, factor=0.2):
	z = None
	if z_input is not None:
		z = z_input
		sample_strokes, m = sample(seq2seq, seq_len=model_params.max_seq_len, temperature=temperature, z=z)
		#print('sample_strokes: ',sample_strokes)
		#print('m: ', m)
		strokes = to_normal_strokes(sample_strokes)
		if draw_mode:
			draw_strokes(strokes, factor, padding = 10)
	return strokes

parser = argparse.ArgumentParser()
parser.add_argument('--data_base_dir', '-db', type=str, default='datasets', help="set the data base dir")
parser.add_argument('--model_path', '-mp', type=str, help="the path of the checkpoint")
parser.add_argument('--index_list', '-il', type=str, default=None, help="numpy list of desired indexs. (default: %(default)s)")
parser.add_argument('--draw', '-d', action = 'store_false', help="whether to draw strokes with this model. (default: %(default)s)")
parser.add_argument('--images', '-i', type = int, default = 25, help="How many images to draw. (default: %(default)s)")
args = parser.parse_args()

data_dir = args.data_base_dir
model_path = args.model_path
exp_dir = '/'.join(model_path.split('/')[-4:-2])

if not os.path.isdir(exp_dir.split('/')[0]):
	os.mkdir(exp_dir.split('/')[0])
if not os.path.isdir(exp_dir):
	os.mkdir(exp_dir)

index_list = np.load(args.index_list) if args.index_list != None else None

with open(os.path.join('/'.join(model_path.split('/')[:-2]), 'logs', 'model_config.json'), 'r') as f:
	model_params = json.load(f)
	with open(os.path.join(exp_dir, 'model_config.json'), 'w') as g:
		json.dump(model_params, g, indent = True)
[train_set, valid_set, test_set, hps_model] = load_dataset(data_dir, model_params, index_list = index_list, exp_dir = exp_dir)

print(model_params)
seq2seq = Seq2seqModel(model_params)  # build model
seq2seq.load_trained_weights(model_path) # load checkpoint
seq2seq.make_sampling_models()	# build sub models that are used to infuse inputs and probe values of intermediate layers
os.system('cp -r ' + os.path.join('/'.join(model_path.split('/')[:-2]), 'tensorboard') + ' ' + os.path.join(exp_dir, 'tensorboard'))
for key in seq2seq.sample_models.keys():
	print(key)
	model = seq2seq.sample_models[key]
	print(model.summary())
	#if 'sample' not in key and 'decoder' not in key:
	#tf.contrib.saved_model.save_keras_model(model, os.path.join(exp_dir, f'{key}'))
	model.save(os.path.join(exp_dir, f'{key}.h5') )

if args.draw:
	num_png = args.images
	# set numpy output to something sensible
	np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
	# little function that displays vector images and saves them to .svg
	model_params['batch_size'] = 1
	model_params['max_seq_len'] = 1
	[train_set, valid_set, test_set, hps_model] = load_dataset(data_dir, model_params, index_list = index_list, exp_dir = '')

	model_params = DotDict(model_params)  
	# Get a sample drawing from the test set, and render it to .svg
	if not os.path.isdir(os.path.join(exp_dir, 'origin_png')):
		os.mkdir(os.path.join(exp_dir, 'origin_png'))
	idxs = []
	useless_strokes, idx, img = test_set.random_sample()
	imgs = img
	idxs.append(idx)
	print(img)
	png_filename = os.path.join(exp_dir, 'origin_png/%04d.png' % idx)
	'''
	img = Image.fromarray(img[0, :, :, 0], 'L')
	img.save(png_filename, 'PNG')
	'''
	cv2.imwrite(png_filename, img[0])
	for i in range(num_png - 1):
		useless_strokes, idx, img = test_set.random_sample()
		print(img.shape)
		imgs = np.vstack((imgs, img))
		idxs.append(idx)
		png_filename = os.path.join(exp_dir, 'origin_png/%04d.png' % idx)
		'''
		img = Image.fromarray(img[0, :, :, 0], 'L')
		img.save(png_filename, 'PNG')
		'''
		cv2.imwrite(png_filename, img[0])
	imgs = np.array(imgs)
	#print(imgs.shape)
	z = encode(imgs)
	mu, sigma = z
	print(len(z))
	sigma_exp = np.exp(sigma / 2.0)
	batch_z = mu + sigma_exp * np.random.normal(size = sigma_exp.shape, loc = 0.0, scale = 1.0)

	if not os.path.isdir(os.path.join(exp_dir, 'png')):
		os.mkdir(os.path.join(exp_dir, 'png'))
	if not os.path.isdir(os.path.join(exp_dir, 'png_cv')):
		os.mkdir(os.path.join(exp_dir, 'png_cv'))
	for i in range(num_png):
                #print(i)
                z = np.expand_dims(batch_z[i], axis = 0)
                strokes = decode(z, temperature=0.5) # convert z back to drawing at temperature of 0.5
                svg_size, dwg_bytestring = draw_strokes(strokes, svg_filename = os.path.join(exp_dir, 'svg/%04d.svg' % idxs[i]), padding = 10)
                svg2png_v2(dwg_bytestring, svg_size, (model_params['img_W'], model_params['img_H']), png_filename = os.path.join(exp_dir, 'png/%04d.png' % idxs[i]), padding=True)
