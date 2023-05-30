# -*- coding: UTF-8 -*-
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np


def calculate_time(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		print(f"Function {func.__name__} took {(end_time - start_time) * 1000:.2f}ms to run")
		return result
	
	return wrapper


def show_image(time2show):
	def decorator(func):
		def wrapper(*args, **kwargs):
			result = func(*args, **kwargs)
			if isinstance(result, np.ndarray) and len(result.shape) == 2:
				plt.imshow(result, cmap='gray')
			else:
				plt.imshow(result)
			plt.show(block=False)
			time.sleep(time2show)
			plt.close()
			return result
		
		return wrapper
	
	return decorator
