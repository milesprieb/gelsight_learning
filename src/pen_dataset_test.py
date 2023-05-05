import json
import os
import numpy as np

def main():
    root_dir = '/home/rpmdt05/Code/Tacto_good/data/'
    output_path = os.path.join(root_dir, 'pen.json')
    with open(output_path) as f:
        data = json.load(f)
    print(len(data))

    rots = {}
    for dat in data:
        rot = dat['j']
        if rot not in rots:
            rots[rot] = 0
        rots[rot] += 1

    bin = {}
    keys = sorted(rots.keys())
    for i in range(len(keys)):
        bin[i] = keys[i]
        


    # print sorted keys
    print(len(rots.keys()))
    for key in sorted(rots.keys()):
        print(key, rots[key])

    print(bin)

if __name__ == '__main__':
    main()
