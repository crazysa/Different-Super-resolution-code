
# import the necessary packages
import argparse
import time
import cv2
import os
import glob
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to super resolution model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image we want to increase resolution of")
args = vars(ap.parse_args())

models_list = ["models/FSRCNN_x4.pb", "models/EDSR_x4.pb", "models/ESPCN_x4.pb", "models/LapSRN_x8.pb"]
images = glob.glob(args["image"] + "/*.jpg")

for image_path in images:
	image = cv2.imread(image_path)
	# print("The Image is", image)
	for model in models_list:
		# extract the model name and model scale from the file path
		modelName = model.split(os.path.sep)[-1].split("_")[0].lower()
		modelScale = model.split("_x")[-1]
		modelScale = int(modelScale[:modelScale.find(".")])

		# initialize OpenCV's super resolution DNN object, load the super
		# resolution model from disk, and set the model name and scale
		print("[INFO] loading super resolution model: {}".format(
			model))
		print("[INFO] model name: {}".format(modelName))
		print("[INFO] model scale: {}".format(modelScale))
		sr = cv2.dnn_superres.DnnSuperResImpl_create()
		sr.readModel(model)
		sr.setModel(modelName, modelScale)

		# load the input image from disk and display its spatial dimensions

		print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
		# use the super resolution model to upscale the image, timing how
		# long it takes
		start = time.time()
		try:
			upscaled = sr.upsample(image)
		except:
			print("The superresolution didn't work for", modelName, image_path)
			continue

		end = time.time()
		print("[INFO] super resolution took {:.6f} seconds".format(
			end - start))
		# show the spatial dimensions of the super resolution image
		print("[INFO] w: {}, h: {}".format(upscaled.shape[1],
			upscaled.shape[0]))


		# resize the image using standard bicubic interpolation
		start = time.time()
		bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]),
			interpolation=cv2.INTER_CUBIC)
		end = time.time()
		print("[INFO] bicubic interpolation took {:.6f} seconds".format(
			end - start))
		bicubic_path = "/home/deepen/Desktop/Shubham"+"/bicubic/"+image_path.split('/')[-1]
		model_path = "/home/deepen/Desktop/Shubham/"+model.split('.')[0] + "/" + image_path.split('/')[-1]
		print("images saved at", bicubic_path, model_path)
		cv2.imwrite(bicubic_path, bicubic)
		cv2.imwrite(model_path, upscaled)

		# show the original input image, bicubic interpolation image, and
		# super resolution deep learning output
		image_resized = cv2.resize (image, (960,540))
		bicubic_resized = cv2.resize(bicubic, (960, 540))
		upscaled_resized = cv2.resize(upscaled, (960,540))
		# cv2.imshow("Original", image_resized)
		# cv2.imshow("Bicubic", bicubic_resized)
		# cv2.imshow("Super Resolution", upscaled_resized)
		# cv2.waitKey(2000)