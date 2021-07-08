# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py

import os


dataset_path = '/workspace/mnt/storage/hourenzheng/kinetics_images/kinetcis_frames/'
label_path = 'data/kinetics400'

if __name__ == '__main__':
    with open('tools/kinetics-400-map.txt') as f:
        categories = f.readlines()
        categories = [c.strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '') for c in categories]
    assert len(set(categories)) == 400
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    print(dict_categories)

    files_input = ['kinetics-400-val.txt', 'kinetics-400-train.txt']#['kinetics_val.csv', 'kinetics_train.csv']
    files_output = ['new_val_videofolder.txt', 'new_train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        count_cat = {k: 0 for k in dict_categories.keys()}

        # get ture labels
        with open(os.path.join(label_path, filename_input)) as f1:
            s1 = f1.readlines()
        folders = []
        idx_categories = []
        categories_list = []
        for s2 in s1:
            train = s2.rstrip().split(' ')[0].split('/')[-1].split('.')[0]
            train_label = s2.rstrip().split(' ')[0].split('/')[-2].strip().replace(' ', '_').replace('"', '').replace('(', '').replace(')', '').replace("'", '')
            folders.append(train)
            categories_list.append(train_label)
            idx_categories.append(dict_categories[train_label])


        assert len(idx_categories) == len(folders)
        missing_folders = []
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            img_dir = os.path.join(dataset_path, categories_list[i], curFolder)
            if not os.path.exists(img_dir):
                missing_folders.append(img_dir)
                # print(missing_folders)
            else:
                dir_files = os.listdir(img_dir)
                output.append('%s %d %d'%(os.path.join(categories_list[i], curFolder), len(dir_files), curIDX))
            print('%d/%d, missing %d'%(i, len(folders), len(missing_folders)))
        with open(os.path.join(label_path, filename_output),'w') as f:
            f.write('\n'.join(output))
        with open(os.path.join(label_path, 'missing_' + filename_output),'w') as f:
            f.write('\n'.join(missing_folders))