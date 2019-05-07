import os
import pandas as pd
import numpy as np
import gin
import shutil
from sklearn.model_selection import train_test_split

@gin.configurable
class FilePrep:
    def __init__(self, exp_directory=None, image_directory=None, remake_train_test=False, train=0.60,
                 labels_csv_path=None, file_name_column=None, labels_column=None, image_paths_name='image_paths_col',
                 use_symlinks=None):
        self.image_directory = image_directory
        self.exp_dir = exp_directory
        self.remake_dirs = remake_train_test
        self.train = train
        self.labels_csv =  labels_csv_path
        self.file_name_column = file_name_column
        self.data_labels_column = labels_column
        self.image_paths_col = image_paths_name
        self.symlink = use_symlinks

    def _make_class_df(self):
        orig_df = pd.read_csv(self.labels_csv)
        file_list = []
        file_names = []
        file_type = '.png'
        file_name_col = self.file_name_column
        data_label_col = self.data_labels_column
        labels_df = orig_df[[file_name_col, data_label_col]]
        for root, _, files in os.walk(self.image_directory):
            for file in files:
                if file.endswith(file_type):
                    file_names.append(file.replace(file_type, ''))
                    file_list.append(os.path.join(root, file))
        files_df = pd.DataFrame.from_dict({file_name_col: file_names, self.image_paths_col:file_list})
        data = pd.merge(files_df, labels_df, how='inner', on=[file_name_col])
        data = data.drop_duplicates(subset=[file_name_col])
        data_no_id = data.drop(columns=[file_name_col])
        return data_no_id

    def _make_dirs(self, paths):
        assert isinstance(paths, list), 'Only make directories from lists, not {}'.format(type(paths))
        for path in paths:
            if self.remake_dirs:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    os.makedirs(path)
                else:
                    os.makedirs(path)
            else:
                try:
                    os.makedirs(path)
                except OSError:
                    print('Directory {} already exists. To remake set remake_dirs=True.'.format(path))

    def unique_label_names(self, labels):
        if isinstance(labels, pd.Series):
            return np.sort(labels.unique())
        elif isinstance(labels, np.ndarray):
            return np.sort(np.unique(labels))

    def _train_test_val(self, data_labels_df):
        img_train, img_test_val, label_train, label_test_val  = train_test_split(data_labels_df[self.image_paths_col],
                                                                                 data_labels_df[self.data_labels_column],
                                                                                 train_size=self.train,
                                                                                 test_size=1-self.train,
                                                                                 random_state=42)
        img_test, img_val, label_test, label_val = train_test_split(img_test_val, label_test_val,
                                                                    train_size=0.5,
                                                                    random_state=42)
        data_dict = {'img_train':img_train,'label_train':label_train,'img_test':img_test,'label_test':label_test,
                     'img_val':img_val,'label_val':label_val}
        return data_dict

    def create_modeling_dataset(self):
        data_labels = self._make_class_df()
        if data_labels.empty:
            raise ValueError(f'No file names matching those in class dataframe in the directory {self.image_directory}')
        train_base = os.path.join(self.exp_dir , 'train')
        test_base  = os.path.join(self.exp_dir , 'test')
        valid_base = os.path.join(self.exp_dir , 'validate')
        self.label_names = self.unique_label_names(data_labels[self.data_labels_column])
        data_dict = self._train_test_val(data_labels)
        for i, label in enumerate(self.label_names):
            s_label = str(label).strip().replace('/', 'and').replace(' ', '_')
            dir_name = 'class_{0:03d}_{1}'.format(i, s_label)
            train_path = os.path.join(train_base, dir_name)
            test_path  = os.path.join(test_base, dir_name)
            valid_path = os.path.join(valid_base, dir_name)
            path_classes = [train_path, test_path, valid_path]
            self._make_dirs(path_classes)
            train_files =  data_dict['img_train'][data_dict['label_train'] == label].tolist()
            for file in train_files:
                _, f_name = os.path.split(file)
                file_dest = os.path.join(train_path, f_name)
                if not os.path.exists(file_dest):
                    if self.symlink:
                        os.symlink(file, file_dest)
                    else:
                        shutil.copy(file, file_dest)
            test_files =  data_dict['img_test'][data_dict['label_test'] == label].tolist()
            for file in test_files:
                _, f_name = os.path.split(file)
                file_dest = os.path.join(test_path, f_name)
                if not os.path.exists(file_dest):
                    if self.symlink:
                        os.symlink(file, file_dest)
                    else:
                        shutil.copy(file, file_dest)
            valid_files =  data_dict['img_val'][data_dict['label_val'] == label].tolist()
            for file in valid_files:
                _, f_name = os.path.split(file)
                file_dest = os.path.join(valid_path, f_name)
                if not os.path.exists(file_dest):
                    if self.symlink:
                        os.symlink(file, file_dest)
                    else:
                        shutil.copy(file, file_dest)
        print('Directory structure built and train/test/validation files moved to directories.')

