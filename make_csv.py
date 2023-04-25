import pandas as pd
import numpy as np
import os
import re

def main():
    path = 'classification_data/images'
    csv_path = 'classification_data/images.csv'
    df = pd.DataFrame(columns=['image', 'label'])
    map = {'bishop': 0, 
             'king': 1,
             'knight': 2,
             'pawn': 3,
             'queen': 4,
             'rook': 5,
            }
    for i, image in enumerate(os.listdir(path)):
        if image.endswith('.jpg'):
            label = re.split(r'[_\d]', image)[1]
            # print(label)
            # print(i, image, label)
            df.loc[i] = [image, map[label]]
    
    print(df)
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    main()