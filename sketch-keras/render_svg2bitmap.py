import os
import subprocess
import argparse
import numpy as np
from PIL import Image
import cairosvg
import six
import svgwrite
import time
import cv2

import utils
from seq2seqVAE import get_default_hparams

def copy_hparams(model_params):
	return model_params.copy()

def draw_strokes(data, svg_filename, factor=0.2, padding=50):
	"""
	little function that displays vector images and saves them to .svg
	:param data:
	:param factor:
	:param svg_filename:
	:param padding:
	:return:
	"""
	min_x, max_x, min_y, max_y = utils.get_bounds(data, factor)
	dims = (padding + max_x - min_x, padding + max_y - min_y)
	dwg = svgwrite.Drawing(svg_filename, size=dims)
	dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
	lift_pen = 1
	abs_x = int(padding / 2) - min_x
	abs_y = int(padding / 2) - min_y
	p = "M%s, %s " % (abs_x, abs_y)
	# use lowcase for relative position
	command = "m"
	for i in range(len(data)):
		if lift_pen == 1:
			command = "m"
		elif command != "l":
			command = "l"
		else:
			command = ""
		x = float(data[i, 0]) / factor
		y = float(data[i, 1]) / factor
		lift_pen = data[i, 2]
		p += command + str(x) + ", " + str(y) + " "
	the_color = "black"
	stroke_width = 1
	dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
	dwg.save()

	return dims, dwg.tostring()

def abs(x):
	return x if x >= 0 else -x

def pad_image(png_filename, pngsize, version):
	curr_png = Image.open(png_filename).convert('RGB')
	png_curr_w = curr_png.width
	png_curr_h = curr_png.height
	if version == 'v1':
		assert png_curr_w == pngsize[0] or png_curr_h == pngsize[1]
	else:
		if png_curr_w != pngsize[0] and png_curr_h != pngsize[1]:
			print('Not aligned', 'png_curr_w', png_curr_w, 'png_curr_h', png_curr_h)
			#print('wanted', 'png_w', pngsize[0], 'png_h', pngsize[1])

	# resized without opencv
	padded_png = np.zeros(shape=[pngsize[1], pngsize[0], 3], dtype=np.uint8)
	padded_png.fill(255)
	diff_w = abs(png_curr_w - pngsize[0])
	diff_h = abs(png_curr_h - pngsize[1])
	half_w = int(diff_w / 2)
	half_h = int(diff_h / 2)
	#print('diff_w', diff_w, 'diff_h', diff_h)
	if png_curr_w < pngsize[0]:
		if png_curr_h < pngsize[1]:
			padded_png[half_h:-diff_h + half_h, half_w:-diff_w + half_w] = np.array(curr_png, dtype = np.uint8)
		elif png_curr_h == pngsize[1]:
			padded_png[:, half_w:-diff_w + half_w] = np.array(curr_png, dtype = np.uint8)
		else:
			padded_png[:, half_w:-diff_w + half_w] = np.array(curr_png, dtype = np.uint8)[half_h:-diff_h + half_h, :]
	elif png_curr_w == pngsize[0]:
		if png_curr_h < pngsize[1]:
			padded_png[half_h:-diff_h + half_h, :] = np.array(curr_png, dtype = np.uint8)
		elif png_curr_h == pngsize[1]:
			padded_png = np.array(curr_png, dtype = np.uint8)
		else:
			padded_png = np.array(curr_png, dtype = np.uint8)[half_h:-diff_h + half_h, :]
	else:
		if png_curr_h < pngsize[1]:
			padded_png[half_h:-diff_h + half_h, :] = np.array(curr_png, dtype = np.uint8)[:, half_w:-diff_w + half_w]
		elif png_curr_h == pngsize[1]:
			padded_png = np.array(curr_png, dtype = np.uint8)[:, half_w:-diff_w + half_w]
		else:
			padded_png = np.array(curr_png, dtype = np.uint8)[half_h:-diff_h + half_h, half_w:-diff_w + half_w]
	
	padded_png = Image.fromarray(padded_png, 'RGB')
	padded_png.save(png_filename, 'PNG')
	
	# resized with opencv
	png_cv = np.array(curr_png, dtype = np.uint8).copy()
	png_cv = cv2.resize(png_cv, (pngsize[1], pngsize[0]))
	png_cv = Image.fromarray(png_cv, 'RGB')
	png_cv.save('/'.join(png_filename.split('/')[:-1]) + '_cv/' + png_filename.split('/')[-1], 'PNG')


def svg2png_v1(input_path, svgsize, pngsize, png_filename, padding=False, padding_args="--export-area-drawing"):
	"""convert svg into png, using inkscape"""
	svg_w, svg_h = svgsize
	png_w, png_h = pngsize
	x_scale = png_w / svg_w
	y_scale = png_h / svg_h

	if x_scale > y_scale:
		y = int(png_h)
		cmd = "inkscape {0} {1} -e {2} -h {3}".format(input_path, padding_args, png_filename, y)
	else:
		x = int(png_w)
		cmd = "inkscape {0} {1} -e {2} -w {3}".format(input_path, padding_args, png_filename, x)

	# Do the actual rendering
	subprocess.call(cmd.split(), shell=False)

	if padding:
		pad_image(png_filename, pngsize, 'v1')


def svg2png_v2(dwg_string, svgsize, pngsize, png_filename, padding=False):
	"""convert svg into png, using cairosvg"""
	svg_w, svg_h = svgsize
	png_w, png_h = pngsize
	x_scale = png_w / svg_w
	y_scale = png_h / svg_h

	if x_scale > y_scale:
		cairosvg.svg2png(bytestring=dwg_string, write_to=png_filename, output_height=png_h)
	else:
		cairosvg.svg2png(bytestring=dwg_string, write_to=png_filename, output_width=png_w)

	if padding:
		pad_image(png_filename, pngsize, 'v2')


def main(**kwargs):
	data_base_dir = kwargs['data_base_dir']
	render_mode = kwargs['render_mode']

	npz_dir = os.path.join(data_base_dir, 'npz')
	svg_dir = os.path.join(data_base_dir, 'svg')
	png_dir = os.path.join(data_base_dir, 'png')

	model_params = get_default_hparams()
	for dataset_i in range(len(model_params['data_set'])):
		assert model_params['data_set'][dataset_i][-4:] == '.npz'
		cate_svg_dir = os.path.join(svg_dir, model_params['data_set'][dataset_i][:-4])
		cate_png_dir = os.path.join(png_dir, model_params['data_set'][dataset_i][:-4])
		
		index_list = np.load(kwargs['index_list']) if kwargs['index_list'] != None else None
		model_params['opencv'] = False
		model_params['stroke_resize'] = False
		datasets = utils.load_dataset(data_base_dir, model_params, index_list = index_list, exp_dir = '')

		data_types = ['train', 'valid', 'test']
		for d_i, data_type in enumerate(data_types):
			split_cate_svg_dir = os.path.join(cate_svg_dir, data_type)
			split_cate_png_dir = os.path.join(cate_png_dir, data_type,
											  str(model_params['img_H']) + 'x' + str(model_params['img_W']))

			split_cate_png_cv_dir = os.path.join(cate_png_dir, data_type, str(model_params['img_H']) + 'x' + str(model_params['img_W']) + '_cv')
			os.makedirs(split_cate_svg_dir, exist_ok=True)
			os.makedirs(split_cate_png_dir, exist_ok=True)
			os.makedirs(split_cate_png_cv_dir, exist_ok=True)

			split_dataset = datasets[d_i]

			for ex_idx in range(len(split_dataset.strokes)):
				stroke = np.copy(split_dataset.strokes[ex_idx])
				print('example_idx', ex_idx, 'stroke.shape', stroke.shape)

				png_path = split_dataset.png_paths[ex_idx]
				png_path.replace('_cv', '')
				assert split_cate_png_dir == png_path[:len(split_cate_png_dir)]
				actual_idx = png_path[len(split_cate_png_dir) + 1:-4]
				svg_path = os.path.join(split_cate_svg_dir, str(actual_idx) + '.svg')

				svg_size, dwg_bytestring = draw_strokes(stroke, svg_path, padding=10)  # (w, h)

				if render_mode == 'v1':
					svg2png_v1(svg_path, svg_size, (model_params['img_W'], model_params['img_H']), png_path, padding=True)
				elif render_mode == 'v2':
					svg2png_v2(dwg_bytestring, svg_size, (model_params['img_W'], model_params['img_H']), png_path, padding=True)
				else:
					raise Exception('Error: unknown rendering mode.')


if __name__ == '__main__':
	start = time.time()
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_base_dir', '-db', type=str, default='datasets', help="set the data base dir")
	parser.add_argument('--render_mode', '-rm', type=str, choices=['v1', 'v2'], default='v1', help="choose a rendering mode")
	parser.add_argument('--index_list', '-il', type=str, default=None, help="numpy list of desired indexs")
	args = parser.parse_args()

	run_params = {
		"data_base_dir": args.data_base_dir,
		"render_mode": args.render_mode,
		"index_list": args.index_list
	}

	main(**run_params)
	print(time.time() - start)
