# -*- coding: UTF-8 -*-
import os

import requests
from bs4 import BeautifulSoup


def download_barcode_images(url, save_directory, num_images):
	# 创建保存图片的目录
	if not os.path.exists(save_directory):
		os.makedirs(save_directory)
	
	count = 0
	try:
		# 发送HTTP GET请求获取图片
		response1 = requests.get(url)
		
		if response1.status_code == 200:
			# 使用Beautiful Soup解析网页内容
			soup = BeautifulSoup(response1.text, 'html.parser')
			# 查找所有的<img>标签
			img_tags = soup.find_all('a')
			# 提取图片链接
			# image_urls = [img['src'] for img in img_tags]
			image_urls = []
			
			for img in img_tags:
				try:
					image_urls.append(img['m']['murl'])
				except Exception as e:
					print(e)
					pass
			
			for image_url in image_urls:
				try:
					# 发送HTTP GET请求获取图片
					response = requests.get(image_url)
					
					if response.status_code == 200:
						# 从URL中提取文件名
						filename = os.path.join(save_directory, f"barcode_{count}.jpg")
						content = response.content
						# 保存图片到本地
						with open(filename, 'wb') as f:
							f.write(content)
						
						count += 1
						print(f"Downloaded image {count}/{num_images}")
				except requests.exceptions.RequestException as e:
					print(f"Error: {str(e)}")
					pass
				finally:
					if count >= num_images:
						break
	except requests.exceptions.RequestException as e:
		print(f"Error: {str(e)}")
		pass


url = "https://www.bing.com/images/search?q=%E5%B7%A5%E4%B8%9A%E5%9C%BA%E6%99%AF%20%E6%9D%A1%E7%A0%81%E6%A0%87%E7%AD%BE&qs=n&form=QBIR&sp=-1&lq=0&pq=%E5%B7%A5%E4%B8%9A%E5%9C%BA%E6%99%AF%20%E6%9C%89%E4%B8%80%E4%B8%AA%E6%88%96%E5%A4%9A%E4%B8%AA%E6%9D%A1%E7%A0%81%E6%A0%87%E7%AD%BE&sc=0-15&cvid=0713E62964D04D1081D26D3B0A190C68&ghsh=0&ghacc=0&first=1"  # 替换为实际的图片URL
save_directory = r"D:\Fenkx\Fenkx - General\AI\Dataset\01"  # 图片保存目录
num_images = 3  # 要下载的图片数量

download_barcode_images(url, save_directory, num_images)
