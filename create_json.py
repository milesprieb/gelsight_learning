import pandas as pd
import json
import os

def create_label_json(filename):
    dirs = os.listdir(filename)
    # df = pd.DataFrame(columns=['Depth_image', 'Label'])
    out = []
    # print(dirs)
    for dir in dirs:
        inner_dir = os.path.join(filename, dir)
        # print(inner_dir)
        files = os.listdir(inner_dir)
        # print(files)
        for file in files:
            out.append({'Depth_image': file, 'Label': dir})
        # print(df)
    print(out)
    json_object = json.dumps(out, indent = 0)
    with open('real_depth/gs_data/king.json', 'w') as outfile:
        outfile.write(json_object)

def main():
    # df = pd.read_csv('classification_data/images.csv')
    # # print(df)
    # out = []
    # for i in range(len(df)):
    # qi for angle
    #     out.append({'RGB_image': df['image'][i]})
    # # print(out)
    # json_object = json.dumps(out, indent = 0)
    # with open('classification_data/king.json', 'w') as outfile:
    #     outfile.write(json_object)
    create_label_json('real_depth/gs_data')

if __name__ == '__main__':
    main()