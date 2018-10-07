import sys
sys.path.append('..')
import json


def dataset_build(config):

    raw_files = [config['train_json'], config['val_json'], config['test_json']]
    processed_files = [config['train_data'],config['val_data'],config['test_data']]

    for i in range(3):
        raw_file = raw_files[i]
        processed_file = processed_files[i]

        processed_data = list()

        with open(raw_file, 'r') as fr:
            raw_data = json.load(fr)
        for vid, data in raw_data.items():
            duration = data['duration']
            timestamps = data['timestamps']
            sentences = data['sentences']
            for idx in range(len(timestamps)):
                processed_data.append([vid,duration,timestamps[idx],sentences[idx]])
        print('sample num:', len(processed_data))
        json.dump(processed_data,open(processed_file,'w'))



if __name__ == '__main__':

    config_file = '../configs/config_base.json'

    with open(config_file, 'r') as fr:
        config = json.load(fr)

    dataset_build(config)
