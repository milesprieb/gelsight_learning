import json
import re 
import pandas as pd

# with open('data') as f:
#     data = json.load(f)
#     print(len(data))
#     print(data.keys())
#     # print(re.split(r'[_\d]', data[1000]['RGB_image']))

df_king = pd.read_json('data/king.json')
df_bishop = pd.read_json('data/bishop.json')
df_knight = pd.read_json('data/knight.json')
df_pawn = pd.read_json('data/pawn.json')
df_queen = pd.read_json('data/queen.json') 
df_rook = pd.read_json('data/rook.json')
df = pd.concat([df_king, df_bishop, df_knight, df_pawn, df_queen, df_rook])
df = df.reset_index(drop=True)
df.to_json('out.json', orient='records')
