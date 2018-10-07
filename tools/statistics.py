import sys
sys.path.append('..')
import h5py
import json
import os


def dataset_distribution(filepath):

    all_frames = 0
    dist_set = dict()

    files = os.listdir(filepath)
    files_num = len(files)
    count = 0
    for file in files:
        filename = os.path.join(filepath, file)
        count += 1
        if count%100 == 0:
            print('read ', count)
        with h5py.File(filename, 'r') as item:
            length = item['feature'].value.shape[0]
            all_frames += length
            if length not in dist_set:
                dist_set[length] = 1
            else:
                dist_set[length] += 1
    print('average frames:', all_frames/files_num)
    print('dist:', dist_set)



if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    dataset_distribution(config["features"])
