import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import open3d as o3d
#from open3d import JVisualizer
from utils import get_block
from enum import Enum
from clean_up_disparity import filter_map



def crop(img, window_size):
	'''
	Parameters:
		img-- numpy array representing a picture
		window_size-- window size used during depth map generation
	Returns:
		img cropped removing window_size/2 pixels from each side of the image
	'''
	shape = img.shape
	h = img.shape[0]
	w = img.shape[1]
	half_window_size = int(window_size/2)
	return img[half_window_size:h-half_window_size, half_window_size:w-half_window_size]

def sad(img1, img2):
	'''
	Parameters:
		img1-- numpy array representing a picture
		img2-- numpy array representing a picture
	Returns:
		float -- sum of absolute differences between img1 and shifted img2
	'''

	return np.sum(np.abs(np.subtract(img1, img2, dtype=float)))

def distance_to_best_block(block1, block1_coordinates, img2, search_size, half_window_size):
	'''
	Parameters:
		block1-- 3 channel (row, col, colors) numpy array representing a block of a picture
		block1_coordinates-- tuple(r, w) or (y, x) representing location of center of block1 (used to calculate distance)
		img2-- 3 channel (row, col, colors) numpy array representing a picture
		search_size-- maximum number of pixels away we can look for matching blocks in img2
		half_window_size-- half size of possible blocks
	Returns:
		float distance between center of block1 and the best matching block within search_size

	iterate through all blocks of (2 * window_size + 1) in img2 no further than search_size away
	find the block with the minimum SAD (sum of absolute differences) to block 1 and retain its location coordinates
	return the distance between block 1 and the best block.
	'''
	[y, block1_x] = block1_coordinates
	
	best_sad = float('inf')
	best_x = block1_x

	for x in range(max(half_window_size, block1_x - search_size), min(img2.shape[1] - half_window_size, block1_x + search_size)):

		block2 = get_block(img2, y, x, half_window_size)

		curr_sad = sad(block1, block2)
		if(curr_sad < best_sad):
			best_sad=curr_sad
			best_x = x
			best_block = block2
	dist = abs(block1_x - best_x)

	return dist

def disparity_map(im_left, im_right, window_size, search_size):
	'''
	Parameters:
		im_left-- left stereo pair image file (numpy array)
		im_right-- right stereo pair image file (numpy array)
		window_size-- half size of possible blocks
		search_size-- maximum number of pixels away we can look for matching blocks in img2

	Returns:
		matrix containing displacement between xl and xr for a pixel (xl - xr)
	'''

	shape = im_left.shape

	h = shape[0]
	w = shape[1]
	
	disparity = np.zeros((h, w), 'uint8');
	half_window_size = int(window_size/2);

	print("creating disparity map...")
	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			block = get_block(im_left, y, x, half_window_size)
			distance = float(distance_to_best_block(block, (y, x), im_right, search_size, half_window_size))
			disparity[y, x] = distance
	print("created disparity map!")

	return crop(disparity, window_size = window_size)

def disparity_to_visible_disparity_map(disparity_matrix, search_size):
	'''
	Parameters:
		disparity_matrix-- matrix containing displacement between xl and xr for a pixel (xl - xr)
		search_size-- maximum pixels away travelled when searching for best matching block
	Returns:
		depth map in mm	
	'''
	disparity_matrix = disparity_matrix.astype(np.float64) / search_size
	disparity_matrix = 255 * disparity_matrix # Now scale by 255
	disparity_matrix = disparity_matrix.astype(np.uint8)
	return disparity_matrix


def disparity_to_depth_map(disparity_matrix, f, t, scale):
	'''
	Parameters:
		disparity_matrix-- matrix containing displacement between xl and xr for a pixel (xl - xr)
		f-- focal length in pixels
		t-- baseline in mm
		scale-- amount origional image was scaled down, used to scale down f and t
	Returns:
		depth map in mm	
	'''

	disparity_matrix = disparity_matrix.astype(np.float64)
	disparity_matrix = (f/scale) * (t/scale) / disparity_matrix
	disparity_matrix = disparity_matrix.astype(np.uint16)
	return disparity_matrix

def display_depth_map(depth_map, color_img, fx, fy, cx, cy, scale):
	'''
	Parameters:
		fx-- focal length in x dir (scaled if resized)
		fy-- focal length in y dir (scaled if resized)
		cx-- x axis principle point (scaled if resized)
		cy-- y axis principle point (scaled if resized)

	Displays an Open3D point cloud
	'''

	fx = fx/scale
	fy = fy/scale
	cx = cx/scale
	cy = cy/scale

	shape = color_img.shape;
	h = shape[0]
	w = shape[1]
	
	img = o3d.geometry.Image(color_img.astype('uint8'))
	depth = o3d.geometry.Image(depth_map.astype('uint16'))

	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)

	o3d_pinhole = o3d.camera.PinholeCameraIntrinsic()
	o3d_pinhole.set_intrinsics(w, h, fx, fy, cx, cy)

	pcd_from_depth_map = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pinhole)
	pcd_from_depth_map.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	o3d.visualization.draw_geometries([pcd_from_depth_map])

class ColorSpace(Enum):
	GREY = 1,
	BGR=2,
	R=3,
	G=4,
	B=5,
	LAB=6,
	L=7,
	YUV=8,
	Y=9,
	RGB=10

def get_data_in_color_space(file, color_space):
	'''
	Parameters:
		file-- file name of image to fetch pixel values for
		color_space-- ColorSpace enum type representing color space to return img as
	Returns:
		numpy array representing image converted into color space defined
	'''
	im_bgr = cv2.imread(file)
	im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
	im_yuv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2YUV)

	if(color_space == ColorSpace.GREY):
		print("fetching " + file + " in GREY")
		return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
	elif(color_space == ColorSpace.R):
		print("fetching " + file + " in R")
		return im_bgr[:,:,2]
	elif(color_space == ColorSpace.G):
		print("fetching " + file + " in G")
		return im_bgr[:,:,1]
	elif(color_space == ColorSpace.B):
		print("fetching " + file + " in B")
		return im_bgr[:,:,0]
	elif(color_space == ColorSpace.LAB):
		print("fetching " + file + " in LAB")
		return im_lab
	elif(color_space == ColorSpace.L):
		print("fetching " + file + " in L")
		return im_lab[:,:,0]
	elif(color_space == ColorSpace.YUV):
		print("fetching " + file + " in YUV")
		return im_yuv
	elif(color_space == ColorSpace.Y):
		print("fetching " + file + " in Y")
		return im_yuv[:,:,0]
	elif(color_space == ColorSpace.RGB):
		print("fetching " + file + " in RGB")
		return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
	else:
		print("fetching " + file + " in BGR")
		return im_bgr

def average_disparity_maps(list_of_maps):
	disparity = np.zeros(list_of_maps[0].shape, 'uint8');
	print(disparity.shape)
	for i in range(list_of_maps[0].shape[0]):
		for j in range(list_of_maps[0].shape[1]):
			disparity_ij = []
			for disparity_matrix in list_of_maps:
				disparity_ij.append(disparity_matrix[i, j])
			disparity[i, j] = sum(disparity_ij)/len(disparity_ij)
	print(disparity)
	return disparity


###### BELOW THIS POINT #####
#### code to actually call methods to create disparity maps ####


img1 = "/Users/nitaishah/Desktop/Stereo-Rectification/rectified_3.png"
img2 = "/Users/nitaishah/Desktop/Stereo-Rectification/rectified_4.png"



window_size = 15
search_size = 100

#left = get_data_in_color_space(img1, ColorSpace.YUV)
#right = get_data_in_color_space(img2, ColorSpace.YUV)

#yuv_disparity_map = disparity_map(left, right, window_size, search_size)

#plt.imshow(yuv_disparity_map, cmap='Greys_r')
#plt.show()

#left = get_data_in_color_space(img1, ColorSpace.GREY)
#right = get_data_in_color_space(img2, ColorSpace.GREY)

grey_disparity_map = disparity_map(img1, img2, window_size, search_size)

plt.imshow(grey_disparity_map, cmap='Greys_r')
plt.show()


#rgb_disparity_map = disparity_map(left, right, window_size, search_size)

#plt.imshow(rgb_disparity_map, cmap='Greys_r')
#plt.show()

final_disparity_map = grey_disparity_map

visible_disparity_map = disparity_to_visible_disparity_map(final_disparity_map, search_size)

print("displaying average disparity map...")
plt.imshow(visible_disparity_map, cmap='Greys_r')
plt.show()

left_gray = crop(get_data_in_color_space(img1, ColorSpace.GREY), window_size = window_size)

filtered_disparity_map = filter_map(final_disparity_map, visible_disparity_map, left_gray, window_size);
visible_filtered_disparity_map = disparity_to_visible_disparity_map(filtered_disparity_map, search_size)

print("displaying filtered yuv disparity map...")
plt.imshow(visible_filtered_disparity_map, cmap='Greys_r')
plt.show()

depth_map = disparity_to_depth_map(filtered_disparity_map, f=16, t=32830, scale=1)
print("created depth map with range: ")
print(np.min(depth_map, axis=None))
print(np.max(depth_map, axis=None))

left_rgb = crop(get_data_in_color_space(img1, ColorSpace.RGB), window_size = window_size)

print("displaying depth map as point cloud!")
display_depth_map(depth_map, left_rgb, fx = 16, fy = 16, cx = 2000, cy = 3000, scale = 1)



