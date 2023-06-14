# -*- coding: UTF-8 -*-
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np


class Timer:
	"""
	定义了一个名为Timer的上下文管理器类。
	在上下文管理器类中，__enter__方法被调用时记录了代码块的开始时间，而__exit__方法在代码块执行完毕后被调用，计算并输出代码块的执行时间。
	示例函数
		def my_function():
			with Timer():
				time.sleep(2)  # 假设这是需要计时的代码
	在my_function函数中，我们使用with Timer():语句来创建一个Timer的实例，并将需要计时的代码块放在with语句块中。
	当程序执行到with Timer():时，上下文管理器的__enter__方法会被调用，记录开始时间；当代码块执行完毕后，__exit__方法会被调用，计算并输出代码块的执行时间。
	"""
	
	def __enter__(self):
		"""
		执行进入代码块之前的操作
		:return:
		"""
		self.start_time = time.perf_counter_ns()
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		"""
		执行退出代码块时的操作
		:param exc_type: 异常的类型。如果在代码块中没有引发异常，则该参数为 None。
		:param exc_val: 异常的实例。如果没有异常发生，则该参数为 None。
		:param exc_tb: 追溯信息（traceback）。它是一个追溯对象，提供了关于异常发生位置和堆栈跟踪的详细信息。你可以使用该参数来记录异常信息、调试代码或进行其他与异常相关的操作。
		:return:
		"""
		end_time = time.perf_counter_ns()
		execution_time = (end_time - self.start_time) / 1000000
		print(f"代码块的运行时间为：{execution_time} 毫秒")


def calculate_time(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.perf_counter_ns()
		result = func(*args, **kwargs)
		end_time = time.perf_counter_ns()
		print(f"Function {func.__name__} took {(end_time - start_time) / 1000000:.3f}ms to run")
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
