import cv2
import numpy as np
import os
import threading
from queue import Queue


def pic_resize(img_path, save_path):
	img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
	
	long_edge = max(img.shape[:2])
	ratio = 1024.0 / long_edge
	img_resized = cv2.resize(img, None, None, ratio, ratio)
	img_resized_shape = img_resized.shape[:2]
	
	if img_resized_shape[0] > img_resized_shape[1]:
		img_padding = cv2.copyMakeBorder(img_resized, 0, 0, 0, (1024 - img_resized_shape[1]), cv2.BORDER_REFLECT)
	elif img_resized_shape[0] < img_resized_shape[1]:
		img_padding = cv2.copyMakeBorder(img_resized, 0, (1024 - img_resized_shape[0]), 0, 0, cv2.BORDER_REFLECT)
	else:
		img_padding = img_resized
	cv2.imwrite(save_path, img_padding)


def worker():
	while True:
		path = q.get()
		pic_resize(path[0], path[1])
		q.task_done()


# Create the queue and start the threads
q = Queue()
for i in range(4):  # Number of threads
	t = threading.Thread(target=worker)
	t.daemon = True
	t.start()

# Add the images to the queue
img_dir = r'D:\Fenkx\Fenkx - General\AI\Dataset\Nature'
save_dir = r'D:\Fenkx\Fenkx - General\AI\Dataset\Nature_Resized'
for img_name in os.listdir(img_dir):
	img_path = os.path.join(img_dir, img_name)
	save_path = os.path.join(save_dir, img_name)
	q.put((img_path, save_path))

# Wait for the queue to be empty
q.join()
