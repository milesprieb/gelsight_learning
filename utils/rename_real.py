import os

def rename_depth_real(filename):
    dirs = os.listdir(filename)
    for dir in dirs:
    # dir = dirs[0]
        inner_dir = os.path.join(filename, dir)
        # print(inner_dir)
        files = os.listdir(inner_dir)
        # print(files, dir)
        for file in files:
            # print(file)
            new_name = 'Depth_' + dir + '_' + file.split('.')[0] + '_' + file.split('.')[1] + '.png'
            # print(new_name)
            os.rename(os.path.join(inner_dir, file), os.path.join(inner_dir, new_name))


def main():
    pass
    # os.chdir('classification_data/images')
    # # path = 'classification_data/images'
    # i = 0
    # for file in os.listdir('.'):
    #     if file.endswith('.jpg'):
    #         print(file)
    #         temp = file.split('.')

    #         # temp[0] = 'RGB'
    #         print(temp)
    #         # final = '_'.join(temp[0:2])
    #         real_final = temp[0] + '.jpg'
    #         print(real_final)
    #         os.rename(file, real_final)
    #         i += 1
    #Depth_rook5243.tif
    # rename_depth_real('real_depth/gs_data')
if __name__ == '__main__':
    main()