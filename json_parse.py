import json
import re 

with open('tacto_dataset/temp/king.json') as f:
    data = json.load(f)
    print(len(data))
    print(data.keys())
    # print(re.split(r'[_\d]', data[1000]['RGB_image']))