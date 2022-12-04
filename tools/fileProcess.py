# -*- coding: UTF-8 -*-

import os

dir_path = r'D:\fy.xie\fenx\fenx - General\Reference\Paper\Machine Vision'
filenames = os.listdir(dir_path)
for index, filename in enumerate(filenames):
    file_url = os.path.join(dir_path, filename)
    if os.path.isfile(file_url):
        new_file_name_list = filename.split('.')
        new_file_name_withoutend = max(new_file_name_list, key=len, default=filename)
        new_file_name = new_file_name_withoutend + '.' + new_file_name_list[-1]
        new_file_url = os.path.join(dir_path, new_file_name)
        os.rename(file_url, new_file_url)
print(filenames)
