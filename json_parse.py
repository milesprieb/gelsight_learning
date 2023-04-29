import json
import re 
import pandas as pd
import os

# with open('data') as f:
#     data = json.load(f)
#     print(len(data))
#     print(data.keys())
#     # print(re.split(r'[_\d]', data[1000]['RGB_image']))
def combine_json(path):
    df_king = pd.read_json(os.path.join(path, 'king.json'))
    df_bishop = pd.read_json(os.path.join(path, 'bishop.json'))
    df_knight = pd.read_json(os.path.join(path, 'knight.json'))
    df_pawn = pd.read_json(os.path.join(path, 'pawn.json'))
    df_queen = pd.read_json(os.path.join(path, 'queen.json')) 
    df_rook = pd.read_json(os.path.join(path, 'rook.json'))
    df = pd.concat([df_king, df_bishop, df_knight, df_pawn, df_queen, df_rook])
    df = df.reset_index(drop=True)
    df.to_json(os.path.join(path, 'out.json'), orient='records')
    # print(df)

def check_json(path):
    df = pd.read_json(os.path.join(path, 'out.json'))
    # print(df['i'].value_counts())
    # print(df['j'].value_counts())
    # print(df['k'].value_counts())
    # print(len(df['i'].unique()))
    # print(len(df['j'].unique()))
    # print(len(df['k'].unique()))
    print(df)

def main():
    path = '/home/user/gelsight/data_depth/data_mod'
    # combine_json(path)
    combine_json(path)
    # pass


if __name__ == '__main__':
    main()