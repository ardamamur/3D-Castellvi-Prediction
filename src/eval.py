import os
import sys
import re
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, cohen_kappa_score
import torch
from torch import nn
from utils.environment_settings import env_settings

# Custom module imports
from utils._prepare_data import DataHandler
from utils._get_model import *
from modules.DenseNetModule import DenseNet
from modules.ResNetModule import ResNet
from dataset.VerSe import *

class Eval:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.data_size = (96,78,78) if (opt.classification_type == "right_side" or opt.classification_type == "right_side_binary") else (128,86,136)
        self.num_classes = opt.num_classes
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt = []
        self.preds = []
        self.dilated_preds = []
        self.eroded_preds = []

    # Implementation of get_label_map function broken down into three functions.
    def get_final_class_prediction_mapping(self):
        prediction_mapping = {
            ('1', '1'): '2b',
            ('2', '2'): '3b',
            ('0', '0'): '0',
            ('0', '1'): '2a',
            ('1', '0'): '2a',
            ('0', '2'): '3a',
            ('2', '0'): '3a',
            ('1', '2'): '4',
            ('2', '1'): '4',
        }
        return prediction_mapping

    def get_side_class_prediction_mapping(self):
        prediction_mapping = {
            '0': '0',
            '2a': '2',
            '2b': '2',
            '3a': '3',
            '3b': '3'
        }
        return prediction_mapping

    def get_output_class_prediction_mapping(self):
        prediction_mapping = {
            '2': '3',
            '1': '2',
            '0': '0'
        }
        return prediction_mapping

    def get_label_map(self, map: str = 'final_class'):
        if self.opt.classification_type == "right_side" or self.opt.classification_type == "right_side_binary" or self.opt.classification_type == "both_side":
            if map == 'final_class':
                prediction_mapping = self.get_final_class_prediction_mapping()

            elif map == 'actual_class':
                prediction_mapping = self.get_side_class_prediction_mapping()

            elif map == 'pred_class':
                prediction_mapping = self.get_output_class_prediction_mapping()

            return prediction_mapping
        else:
            raise NotImplementedError("Only right_side classification is supported")

    # Implementation of load_model function broken down into two functions.
    def initialize_model(self, params):
        if params.model == 'densenet':
            model = DenseNet(opt=params, num_classes=self.num_classes, data_size=self.data_size, data_channel=1)
        elif params.model == 'resnet' or params.model == 'resnetc':
            model = ResNet(opt=params, num_classes=self.num_classes)
        else:
            raise NotImplementedError("Model not recognized. Implement the model initialization.")
        return model

    def load_model(self, path, params):
        model = self.initialize_model(params)
        model.load_state_dict(torch.load(path)['state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def get_test_subjects(self):
        # extract the test subjects from masterlist by usıng the Split column and if Flip is 0
        masterlist = pd.read_excel(self.opt.master_list, index_col=0)
        #masterlist = masterlist.loc[masterlist['Split'] == 'test'] # extract test subjects
        masterlist = masterlist[masterlist['Full_Id'].str.contains('|'.join(self.opt.dataset))]
        if self.opt.eval_type != 'all':
            # return the subjects if FullId contains any substring from dataset array 
            print('eval type is not all')
            masterlist = masterlist[masterlist['Split'].isin(['val'])]
        else:
            masterlist = masterlist[masterlist['Split'].isin(['test', 'val', 'train'])]

        masterlist = masterlist.loc[masterlist['Flip'] == 0] # extract subjects with Flip = 0
        test_subjects = masterlist["Full_Id"].tolist()
        return test_subjects
    
    def get_processor(self):
        processor = DataHandler(master_list=self.opt.master_list,
                            dataset=self.opt.data_root,
                            data_types=self.opt.data_types,
                            image_types=self.opt.img_types
                        )
        return processor
    
    def get_records(self, processor, test_subjects):
        test_records = []
        verse_records = processor.verse_records
        tri_records = processor.tri_records
        records = verse_records + tri_records
        for index in range(len(records)):
            record = records[index]
            # check if any element in test subhjects contains the subject in record
            if any(record['subject'] in s for s in test_subjects):
                # check for the flip value
                if record['flip'] == 0:
                    test_records.append(record)
                else:
                    continue
            else:
                continue
        print('lenght of the test dataset:' , len(test_records))
        return test_records
    

    def apply_softmax(self, outputs):
        # apply softmax to get probabilities
        output_probabilities = self.softmax(outputs)
        return output_probabilities
    
    def get_max(self, output_probabilities):
        # get the max probability
        max_prob, max_index = torch.max(output_probabilities, 1)
        output_class = str(max_index.item())
        output_prob = max_prob.item()
        return output_class, output_prob
    

    def get_model_output(self, model, input):
        # get the prediction from the mode
        print('input shape:', input.shape)
        prediction = model(input)
        return prediction
    
    def get_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1
    
    def get_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm


    def get_results_df(self):
        path = self.opt.results_df
        if os.path.exists(path):
            results_df = pd.read_csv(path, index_col=0)
            return results_df
        else:
            # should be a pivot df where we group by subject and each subject has rows for different experiments
            return None

    def dilate_image(self, img, v_index, iterations=2):
        img = ndimage.binary_dilation(img, iterations=iterations)
        img = img * v_index
        return img
    
    def get_dilated_input(self, sacrum_seg, last_l_seg, s_idx, l_idx, iterations=2):
        dilated_sacrum_seg = self.dilate_image(sacrum_seg, s_idx, iterations=iterations)
        dilated_last_l_seg = self.dilate_image(last_l_seg, l_idx, iterations=iterations)
        dilated_img = dilated_sacrum_seg + dilated_last_l_seg
        return dilated_img
    
    def erosion_image(self, img, v_index, iterations=2):
        img = ndimage.binary_erosion(img, iterations=iterations)
        img = img * v_index
        return img
    
    def get_eroded_input(self, sacrum_seg, last_l_seg, s_idx, l_idx, iterations=2):
        eroded_sacrum_seg = self.erosion_image(sacrum_seg, s_idx, iterations=iterations)
        eroded_last_l_seg = self.erosion_image(last_l_seg, l_idx, iterations=iterations)
        eroded_img = eroded_sacrum_seg + eroded_last_l_seg
        return eroded_img
    
    def convert_to_tensor(self, img):
        img = img[np.newaxis,np.newaxis, ...]
        img = img.astype(np.float32) 
        img = torch.from_numpy(img)
        img = img.float()
        img = img.to(self.device)
        return img
    
    def flip_image(self, img):
        img = np.flip(img, axis=2)
        return img
    

    def get_sacrum_segmentation(self, img):
        s_idx = 26
        sacrum_seg = np.where(img == s_idx)
        return sacrum_seg, s_idx
    
    def get_last_lumbar_segmentation(self, img):
        l_idx = 25 if 25 in img else 24 if 24 in img else 23
        last_lumbar_seg = np.where(img == l_idx)
        return last_lumbar_seg, l_idx
    
    def get_model_inputs(self, img):
        """
        input_types: list of strings
                    ["dilation", "erosion", "original"]
        """
        model_inputs = {
            "original": None,
            "dilation": None,
            "erosion": None,
            "flipped": None,
            "flipped_dilation": None,
            "flipped_erosion": None
        }
        original_img = img.copy()
        sacrum_seg, s_idx = self.get_sacrum_segmentation(original_img)
        last_l_seg, l_idx = self.get_last_lumbar_segmentation(original_img)

        flipped_img = self.flip_image(img.copy())
        flipped_sacrum_seg, flipped_s_idx = self.get_sacrum_segmentation(flipped_img)
        flipped_last_l_seg, flipped_l_idx = self.get_last_lumbar_segmentation(flipped_img)

        original_img_tensor = self.convert_to_tensor(original_img)
        flipped_img_tensor = self.convert_to_tensor(flipped_img)

        model_inputs["original"] = original_img_tensor
        model_inputs["flipped"] = flipped_img_tensor

  
        if self.opt.seg_comparison:
            # Get dilated images
            dilated_img = self.get_dilated_input(sacrum_seg, last_l_seg, s_idx, l_idx)
            flipped_dilated_img = self.get_dilated_input(flipped_sacrum_seg, flipped_last_l_seg, flipped_s_idx, flipped_l_idx)
            print(dilated_img.shape, flipped_dilated_img.shape)

            # Get eroded images
            eroded_img = self.get_eroded_input(sacrum_seg, last_l_seg, s_idx, l_idx)
            flipped_eroded_img = self.get_eroded_input(flipped_sacrum_seg, flipped_last_l_seg, flipped_s_idx, flipped_l_idx)

            model_inputs["dilation"] = self.convert_to_tensor(dilated_img)
            model_inputs["erosion"] = self.convert_to_tensor(eroded_img)
            model_inputs["flipped_dilation"] = self.convert_to_tensor(flipped_dilated_img)
            model_inputs["flipped_erosion"] = self.convert_to_tensor(flipped_eroded_img)

        return model_inputs

    def process_input(self, processor, record):

        if self.opt.classification_type == "right_side" or self.opt.classification_type == "right_side_binary":
            img = processor._get_right_side_cutout(record,return_seg=self.opt.use_seg, max_shape=self.data_size)
        else:
            img = processor._get_cutout(record,return_seg=self.opt.use_seg, max_shape=self.data_size)
        
        if self.opt.use_seg:
            if self.opt.use_zero_out:
                l_idx = 25 if 25 in img else 24 if 24 in img else 23
                l_mask = img == l_idx #create a mask for values belonging to lowest L
                sac_mask = img == 26 #Sacrum is always denoted by value of 26
                lsac_mask = (l_mask + sac_mask) != 0
                lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
                img = img * lsac_mask

            if self.opt.use_bin_seg:
                bin_mask = img != 0
                img = bin_mask.astype(float)
                
        elif self.opt.use_zero_out:
            #We need the segmentation mask to create the boolean zero-out mask, TODO: Use seg-subreg mask in future for better details
            if self.opt.classification_type == "right_side" or self.opt.classification_type == "right_side_binary":
                seg = processor._get_right_side_cutout(record, return_seg=self.opt.use_seg, max_shape=self.data_size)
            else:
                seg = processor._get_cutout(record, return_seg=self.opt.use_seg, max_shape=self.data_size) 
            
            l_idx = 25 if 25 in seg else 24 if 24 in seg else 23
            l_mask = seg == l_idx #create a mask for values belonging to lowest L
            sac_mask = seg == 26 #Sacrum is always denoted by value of 26
            lsac_mask = (l_mask + sac_mask) != 0
            lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
            img = img * lsac_mask

        model_inputs = self.get_model_inputs(img)
        return model_inputs
    

    def get_actual_labels(self, record):
        if record['side'] == 'R':
            if record['castellvi'] == '4':
                acutal_r = self.get_label_map(map='actual_class')['3a']
                actual_flip_r = self.get_label_map(map='actual_class')['2a']
            elif record['castellvi'] in ['3a', '2a']:
                acutal_r = self.get_label_map(map='actual_class')[record['castellvi']]
                actual_flip_r = self.get_label_map(map='actual_class')['0']
            else:
                acutal_r = self.get_label_map(map='actual_class')[record['castellvi']]
                actual_flip_r = self.get_label_map(map='actual_class')[record['castellvi']]
        else:
            if record['castellvi'] == '4':
                acutal_r = self.get_label_map(map='actual_class')['2a']
                actual_flip_r = self.get_label_map(map='actual_class')['3a']
            elif record['castellvi'] in ['3a', '2a']:
                acutal_r = self.get_label_map(map='actual_class')['0']
                actual_flip_r = self.get_label_map(map='actual_class')[record['castellvi']]
            else:
                acutal_r = self.get_label_map(map='actual_class')[record['castellvi']]
                actual_flip_r = self.get_label_map(map='actual_class')[record['castellvi']]
        
        return acutal_r, actual_flip_r


    def process_output(self, model, img):
        with torch.no_grad():
            output = self.get_model_output(model, img)
            output_probabilities = self.apply_softmax(output)
            output_class, output_prob = self.get_max(output_probabilities)
            return output_class, output_prob
        
    def create_results(self, record, castellvi_pred, pred_side, output_prob, pred_flip_side, flipped_output_prob, actual_side, actual_flip_side):
        results = {
            'subject' : record['subject'],
            'version' : self.opt.version_no,
            'actual' : record['castellvi'],
            'pred' : castellvi_pred,
            'actual_right_side' : actual_side,
            'pred_right_side' : pred_side,
            'pred_prob' : output_prob,
            'actual_flip_right_side' : actual_flip_side,
            'pred_flip_right_side' : pred_flip_side,
            'pred_flip_prob' : flipped_output_prob
        }
        return results
    

    def update_results_for_seg(self, dilated_output_prob, eroded_output_prob, pred_dilated_side, pred_eroded_side, pred_flip_dilated_side, pred_flip_eroded_side, dilated_castellvi_pred, eroded_castellvi_pred, flipped_dilated_output_prob, flipped_eroded_output_prob):
        return {
            'dilated_pred' : dilated_castellvi_pred,
            'eroded_pred' : eroded_castellvi_pred,
            'dilated_pred_right_side' : pred_dilated_side,
            'eroded_pred_right_side' : pred_eroded_side,
            'pred_prob' : dilated_output_prob,
            'dilated_pred_prob' : dilated_output_prob,
            'eroded_pred_prob' : eroded_output_prob,
            'pred_flip_right_side' : pred_flip_dilated_side,
            'dilated_pred_flip_right_side' : pred_flip_dilated_side,
            'eroded_pred_flip_right_side' : pred_flip_eroded_side,
            'pred_flip_prob' : flipped_dilated_output_prob,
            'dilated_pred_flip_prob' : flipped_dilated_output_prob,
            'eroded_pred_flip_prob' : flipped_eroded_output_prob
        }
    

    def convert_dict_to_dataframe(self, results):
        # Convert dictionary to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def save_results(self, results_df, base_path):
        results_file = base_path + '/results.csv'
        if not os.path.isfile(results_file):
            results_df.to_csv(results_file, index=False)
        else: # else it exists so append without writing the header
            results_df.to_csv(results_file, mode='a', header=False, index=False)


    def evaluate(self, path, base_path):
        model = self.load_model(path, self.opt)
        processor = self.get_processor()
        test_subjects = self.get_test_subjects()
        records = self.get_records(processor, test_subjects)
        results_list = []
        for record in records:
            self.gt.append(record['castellvi'])
            actual_side, actual_flip_side = self.get_actual_labels(record)
            
            # get input image
            model_inputs = self.process_input(processor, record)
            input_img = model_inputs["original"]
            flipped_img = model_inputs["flipped"]
            output_class, output_prob = self.process_output(model, input_img)
            flipped_output_class, flipped_output_prob = self.process_output(model, flipped_img)
            castellvi_pred = self.get_label_map(map='final_class').get((output_class, flipped_output_class))
            self.preds.append(castellvi_pred)
            pred_side = self.get_label_map(map='pred_class').get((output_class))
            pred_flip_side = self.get_label_map(map='pred_class').get((flipped_output_class))
            results = self.create_results(record, 
                                          castellvi_pred, 
                                          pred_side, 
                                          output_prob, 
                                          pred_flip_side, 
                                          flipped_output_prob, 
                                          actual_side, 
                                          actual_flip_side)

            if self.opt.seg_comparison:
                dilated_img = model_inputs["dilation"]
                eroded_img = model_inputs["erosion"]
                flipped_dilated_img = model_inputs["flipped_dilation"]
                flipped_eroded_img = model_inputs["flipped_erosion"]
                dilated_output_class, dilated_output_prob = self.process_output(model, dilated_img)
                flipped_dilated_output_class, flipped_dilated_output_prob = self.process_output(model, flipped_dilated_img)
                eroded_output_class, eroded_output_prob = self.process_output(model, eroded_img)
                flipped_eroded_output_class, flipped_eroded_output_prob = self.process_output(model, flipped_eroded_img)
                dilated_castellvi_pred = self.get_label_map(map='final_class').get((dilated_output_class, flipped_dilated_output_class))
                eroded_castellvi_pred = self.get_label_map(map='final_class').get((eroded_output_class, flipped_eroded_output_class))
                self.dilated_preds.append(dilated_castellvi_pred)
                self.eroded_preds.append(eroded_castellvi_pred)
                pred_dilated_side = self.get_label_map(map='pred_class').get((dilated_output_class))
                pred_flip_dilated_side = self.get_label_map(map='pred_class').get((flipped_dilated_output_class))
                pred_eroded_side = self.get_label_map(map='pred_class').get((eroded_output_class))
                pred_flip_eroded_side = self.get_label_map(map='pred_class').get((flipped_eroded_output_class))
                results.update(self.update_results_for_seg(dilated_output_prob, eroded_output_prob, 
                                                           pred_dilated_side, pred_eroded_side, pred_flip_dilated_side, 
                                                           pred_flip_eroded_side, dilated_castellvi_pred, eroded_castellvi_pred, 
                                                           flipped_dilated_output_prob, flipped_eroded_output_prob))
                
            results_list.append(results)


        results_df = self.convert_dict_to_dataframe(results_list)
        self.save_results(results_df, base_path)

        return self.gt, self.preds


def main(params, ckpt_path=None, base_path=None):
    evaluator = Eval(params)
    best_models= os.listdir(ckpt_path)
    # TODO : parse the file name to get the best model
    # example : densenet-epoch=75-val_mcc=0.89.ckpt
    best_val = 0
    best_model = None
    for model in best_models:
        val_score = model.split('=')[-1].split('.')[0] + '.' + model.split('=')[-1].split('.')[1]
        val_score = float(val_score)
        if val_score > best_val:
            best_val = val_score
            best_model = model
    best_model_path = os.path.join(ckpt_path, best_model)
    print('Best Model Path: ', best_model_path)
    
    # Run evaluation for each best model
    gt, preds = evaluator.evaluate(path=best_model_path, base_path=base_path)
    
    # Calculate Confusion Matrix, F1, Cohen's Kappa and Matthews Correlation Coefficient
    cm = confusion_matrix(gt, preds)
    f1 = evaluator.get_f1_score(gt, preds)
    ck = cohen_kappa_score(gt, preds)
    mcc = matthews_corrcoef(gt, preds)

    print('Confusion Matrix: \n', cm)
    print('F1 Score: ', f1)
    print('Cohen\'s Kappa: ', ck)
    print('Matthews Correlation Coefficient: ', mcc)
    

    # check if the file exists
    metrics_file = base_path + '/metrics.csv'
    if not os.path.isfile(metrics_file):
        # Initialize an empty DataFrame
        metrics_df = pd.DataFrame(columns=['version', 'f1_score','cohen_kappa', 'mcc'])
        
    else: # else it exists so append without writing the header
        metrics_df = pd.read_csv(metrics_file)

    # Append row to the metrics DataFrame
    metrics_df = metrics_df.append({'version': params.version_no, 'f1_score': f1, 'cohen_kappa': ck, 'mcc': mcc}, ignore_index=True)
    metrics_df.to_csv(metrics_file, index=False)


if __name__ == "__main__":

    if env_settings.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env_settings.CUDA_VISIBLE_DEVICES)

    if env_settings.BIDS_PATH is not None:
        sys.path.append(env_settings.BIDS_PATH)

    if torch.cuda.is_available():
        print('Running on GPU #' + str(torch.cuda.current_device()))
    else:
        print('Running on CPU')

    parser = argparse.ArgumentParser(description='Evaluation settings')

    parser = argparse.ArgumentParser(description='Evaluation settings')
    parser.add_argument('--data_root', nargs='+', default=[str(os.path.join(env_settings.DATA, 'dataset-verse19')),
                                                           str(os.path.join(env_settings.DATA, 'dataset-verse20')),
                                                           str(os.path.join(env_settings.DATA, 'dataset-tri'))])
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'])
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'])
    parser.add_argument('--master_list', default= str(os.path.join(env_settings.ROOT, 'src/dataset/Castellvi_list_Final_Split_v2.xlsx')))
    parser.add_argument('--classification_type', default='right_side')
    parser.add_argument('--castellvi_classes', nargs='+', default=['1a', '1b', '2a', '2b', '3a', '3b', '4', '0'])
    parser.add_argument('--model', default='densenet')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--total_iterations', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_intervals', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--experiments', default=env_settings.EXPERIMENTS)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--port', type=int, default=2023)
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--dropout_prob', type=float, default=0.3)


    parser.add_argument('--rotate_range', type=int, default=10)
    parser.add_argument('--shear_range', type=float, default=0.2)
    parser.add_argument('--translate_range', type=float, default=0.15)
    parser.add_argument('--scale_range', nargs='+', default=[0.9, 1.1])
    parser.add_argument('--aug_prob', type=float, default=0.5)


    parser.add_argument('--use_seg', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--weighted_sample', action='store_true')
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--flip_all', action='store_true')
    parser.add_argument('--cross_validation', action='store_true')
    parser.add_argument('--use_bin_seg', action='store_true')
    parser.add_argument('--use_zero_out', action='store_true')
    parser.add_argument('--gradual_freezing', action='store_true')
    parser.add_argument('--elastic_transform', action='store_true')
    parser.add_argument('--seg_comparison', action='store_true')
    
    parser.add_argument('--version_no', type=int, default=0)
    parser.add_argument('--dataset', nargs='+', default=['verse', 'tri'])
    parser.add_argument('--eval_type', type=str, default='test')
    
    params = parser.parse_args()
    base_experiment = params.experiments + '/baseline_models/' + params.model
    ckpt_path = base_experiment + '/best_models/version_' + str(params.version_no)
    print(ckpt_path)
    main(params=params, ckpt_path=ckpt_path, base_path=base_experiment)
