import gin
import math
import os
import pandas as pd
from tqdm import tqdm
from cv_framework.data_access.data_prep import FilePrep
from cv_framework.model_definitions.model_utils import data_shape
from cv_framework.training.model_comp import comp_model
from cv_framework.data_access.generators import directory_flow
from cv_framework.training.train import save_model, fit_generator
from cv_framework.diagnostics import basic_diagnostics
from cv_framework.model_definitions import standard_arch_dict

@gin.configurable
class CompVisExperiment:

    def __init__(self,
                 base_directory=None,
                 image_directory=None,
                 experiment_name=None,
                 labels_csv=None,
                 file_name_column='file_name',
                 labels_column='class',
                 use_symlinks=False):

        exp_dir = os.path.join(base_directory, str(experiment_name))
        if not os.path.exists(exp_dir):
            try:
                os.makedirs(exp_dir)
            except Exception as e:
                print(f'Could not make experimental directory: {e}')

        print("Building Train/Test/Validation data directories.")
        if not image_directory:
            image_directory = os.path.join(exp_dir, 'images')

        labels_csv_path = os.path.join(base_directory, labels_csv)
        file_prep = FilePrep(
            exp_directory=exp_dir,
            image_directory=image_directory,
            labels_csv_path=labels_csv_path,
            file_name_column=file_name_column,
            labels_column=labels_column,
            use_symlinks=use_symlinks)
        file_prep.create_modeling_dataset()

        self.unique_class_labels = file_prep.label_names
        self.train_dir = os.path.join(f'{exp_dir}/train')
        self.test_dir = os.path.join(f'{exp_dir}/test')


    def standard_models(self):
        print(list(standard_arch_dict.standard_dict.keys()))

    def _build_model_dict(self, model_dict):
        model_dictionary = {}
        for arch_name, model_list in model_dict.items():
            for model in model_list:
                model_dictionary[model] = standard_arch_dict.standard_dict[arch_name]
        return model_dictionary

    @gin.configurable
    def build_models(self, model_dict, summary=True):
        model_dictionary = self._build_model_dict(model_dict)
        compiled_models = {}
        for model_name, model_arch in model_dictionary.items():
            with gin.config_scope(model_name):
                _, in_shape, out_shape = data_shape()
                cnn_model = comp_model(
                    model_name=model_name,
                    model_arch=model_arch,
                    input_shape=in_shape,
                    classes=out_shape)
                compiled_models[model_name] = cnn_model
                if summary:
                    compiled_models[model_name].summary()
        return compiled_models

    @gin.configurable
    def train_models(self, train_list, compiled_models, model_type='bin_classifier', save_figs=False,
                     print_class_rep=True):
        score_dict = {}
        for model_name in tqdm(train_list):
            history_dict = {}
            with gin.config_scope(model_name):
                image_size, _, _ = data_shape()
                # print(f'\nModel name: {model_name} \nImage Size: {image_size}')
                train_gen = directory_flow(dir=self.train_dir, shuffle=True, image_size=image_size)
                test_gen = directory_flow(dir=self.test_dir, shuffle=False, image_size=image_size)
                save_name = f'{str(model_name)}.h5'
                history = fit_generator(
                    model_name=model_name,
                    model=compiled_models[model_name],
                    gen=train_gen,
                    validation_data=test_gen)
                save_model(model=compiled_models[model_name], model_name=save_name)
                history_dict[model_name] = history.history
                test_gen.reset()
                preds = compiled_models[model_name].predict_generator(
                    test_gen,
                    verbose=1,
                    steps=math.ceil(len(test_gen.classes)/test_gen.batch_size)
                )
                score_dict[model_name] = self.score_models(
                    preds, model_name,
                    history=history_dict[model_name],
                    save_figs=save_figs,
                    model_type=model_type,
                    print_class_rep=print_class_rep,
                    test_gen=test_gen)

        model_table = pd.DataFrame(score_dict).transpose().reset_index().rename(mapper={'index':'Model_Name'}, axis=1)
        return compiled_models, model_table

    @gin.configurable
    def score_models(self, preds, model, history=None, save_figs=False, model_type=None, print_class_rep=None,
                     test_gen=None):
        if model_type == 'bin_classifier':
            return self._score_binary_classifiers(preds, model, history, save_figs, print_class_rep, test_gen)
        elif model_type == 'multiclass':
            return self._score_multi_classifiers(preds, model, history, save_figs, print_class_rep, test_gen)

    @gin.configurable
    def _score_binary_classifiers(self, preds, model, history, save_figs, print_class_rep, test_gen):
        model = str(model)
        sens, spec, roc_auc, class_rep, TP, TN, FP, FN, PPV, NPV, FPR, FNR = basic_diagnostics.binary_metrics(
            test_gen.classes, preds, history=history, save_figs=save_figs, class_names=self.unique_class_labels,
            model_name=model)
        model_scores = {'Sensitivity':sens, 'Specificity':spec, 'ROC_AUC_SCORE':roc_auc, 'True_Positives':TP,
                        'True_Negatives':TN, 'False_Positives':FP, 'False_Negatives':FN,
                        'Positive_Predictive_Value':PPV, 'Negative_Predictive_Value':NPV, 'False_Positive_Rate':FPR,
                        'False_Negative_Rate':FNR}

        if print_class_rep:
            print(class_rep)

        return model_scores

    @gin.configurable
    def _score_multi_classifiers(self, preds, model, history, save_figs, print_class_rep, test_gen):
        model = str(model)
        class_rep, TP, TN, FP, FN, PPV, NPV, FPR, FNR = basic_diagnostics.multi_metrics(
            test_gen.classes, preds, history=history, save_figs=save_figs, class_names=self.unique_class_labels,
            model_name=model)
        model_scores = {'True_Positives':TP, 'True_Negatives':TN, 'False_Positives':FP, 'False_Negatives':FN,
                        'Positive_Predictive_Value':PPV, 'Negative_Predictive_Value':NPV, 'False_Positive_Rate':FPR,
                        'False_Negative_Rate':FNR}

        if print_class_rep:
            print(class_rep)

        return model_scores



