import os
import sys
import re
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn

# Append path to import custom modules
sys.path.append('/data1/practical-sose23/castellvi/castellvi_prediction/bids')

# Custom module imports
from utils._prepare_data import DataHandler
from utils._get_model import *
from modules.DenseNetModule import DenseNet
from modules.ResNetModule import ResNet
from dataset.VerSe import *

class Eval:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.data_size = (128, 86, 136)
        self.num_classes = opt.num_classes
        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gt = []
        self.preds = []

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
        if self.opt.classification_type == "right_side":
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
            # print('record:', record['subject'])
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
        # get the prediction from the model
        prediction = model(input)
        return prediction
    
    def get_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
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

    def process_input(self, processor, record):
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
            seg = self.processor._get_cutout(record, return_seg=self.use_seg, max_shape=self.pad_size) 
            l_idx = 25 if 25 in seg else 24 if 24 in seg else 23
            l_mask = seg == l_idx #create a mask for values belonging to lowest L
            sac_mask = seg == 26 #Sacrum is always denoted by value of 26
            lsac_mask = (l_mask + sac_mask) != 0
            lsac_mask = ndimage.binary_dilation(lsac_mask, iterations=2)
            img = img * lsac_mask

        
        # l_idx = 25 if 25 in img else 24 if 24 in img else 23
        # # sacrum part
        # sacrum_seg = np.where(img == 26)
        # # dilate the sacrum seg
        # sacrum_dilated_seg = ndimage.binary_dilation(sacrum_seg, iterations=2)
        # sacrum_dilated_seg = sacrum_dilated_seg*26
        # # erosion the sacrum seg
        # sacrum_erosin_seg = ndimage.binary_erosion(sacrum_seg, iterations=2)
        # sacrum_erosin_seg = sacrum_erosin_seg*26
        


        # # last L
        # last_l_seg = np.where(img == l_idx)
        # # dilate the last L seg
        # last_l_dilated_seg = ndimage.binary_dilation(last_l_seg, iterations=2)
        # last_l_dilated_seg = last_l_dilated_seg*l_idx
        # # erosion the last L seg
        # last_l_erosion_seg = ndimage.binary_erosion(last_l_seg, iterations=2)
        # last_l_erosion_seg = last_l_erosion_seg*l_idx

        # # combine dilated and eroded sacrum and last L seg
        # combined_dilated_seg =  last_l_dilated_seg + sacrum_dilated_seg
        # combined_erosion_seg = last_l_erosion_seg + sacrum_erosin_seg


        # flip the image
        flipped_img = np.flip(img, axis=2)

        img = img[np.newaxis,np.newaxis, ...]
        flipped_img = flipped_img[np.newaxis, np.newaxis, ...]
        # combined_dilated_seg = combined_dilated_seg[np.newaxis, np.newaxis, ...]
        # combined_erosion_seg = combined_erosion_seg[np.newaxis, np.newaxis, ...]

        # Convert to tensor
        img = img.astype(np.float32) 
        flipped_img = flipped_img.astype(np.float32)
        # combined_dilated_seg = combined_dilated_seg.astype(np.float32)
        # combined_erosion_seg = combined_erosion_seg.astype(np.float32)  
        img = torch.from_numpy(img)
        flipped_img = torch.from_numpy(flipped_img)
        # combined_dilated_seg = torch.from_numpy(combined_dilated_seg)
        # combined_erosion_seg = torch.from_numpy(combined_erosion_seg)

        # Convert to float
        img = img.float()
        flipped_img = flipped_img.float()
        # combined_dilated_seg = combined_dilated_seg.float()
        # combined_erosion_seg = combined_erosion_seg.float()

        # Move to GPU
        img = img.to(self.device)
        flipped_img = flipped_img.to(self.device)
        # combined_dilated_seg = combined_dilated_seg.to(self.device)
        # combined_erosion_seg = combined_erosion_seg.to(self.device)

        return img, flipped_img #, combined_dilated_seg, combined_erosion_seg
    

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


    def evaluate(self, path):
        model = self.load_model(path, self.opt)
        processor = self.get_processor()
        test_subjects = self.get_test_subjects()
        records = self.get_records(processor, test_subjects)

        for index in range(len(records)):
            record = records[index]
            # get input image
            input_img, flipped_img = self.process_input(processor, record)
            self.gt.append(record['castellvi'])
            actual_side, actual_flip_side = self.get_actual_labels(record)

            with torch.no_grad():
                # get the output from the model
                output = self.get_model_output(model, input_img)
                flipped_output = self.get_model_output(model, flipped_img)
                # dilated_output = self.get_model_output(model, dilated_img)
                # erosion_output = self.get_model_output(model, erosion_img)

                # apply softmax to get probabilities
                output_probabilities = self.apply_softmax(output)
                flipped_output_probabilities = self.apply_softmax(flipped_output)
                
                output_class, output_prob = self.get_max(output_probabilities)
                flipped_output_class, flipped_output_prob = self.get_max(flipped_output_probabilities)

                castellvi_pred = self.get_label_map(map='final_class').get((output_class, flipped_output_class))
                self.preds.append(castellvi_pred)

                pred_side = self.get_label_map(map='pred_class').get((output_class))
                pred_flip_side = self.get_label_map(map='pred_class').get((flipped_output_class))

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

            #print(results)
            # Convert dictionary to DataFrame
            # Initialize an empty DataFrame
            results_df = pd.DataFrame.from_dict(results, orient='index').T
            #print(results_df)
            if not os.path.isfile('results.csv'):
                results_df.to_csv('results.csv', index=False)
            else: # else it exists so append without writing the header
                results_df.to_csv('results.csv', mode='a', header=False, index=False)
        return self.gt, self.preds


def main(params, ckpt_path=None):
    evaluator = Eval(params)
    best_model= os.listdir(ckpt_path)[0]
    best_model_path = os.path.join(ckpt_path, best_model)
    
    # Run evaluation for each best model
    gt, preds = evaluator.evaluate(path=best_model_path)
    
    f1 = evaluator.get_f1_score(gt, preds)
    cm = evaluator.get_confusion_matrix(gt, preds)

    print('Version Number:', params.version_no)
    # print("Model: ", ckpt_path)
    print("F1 score: ", f1)
    print("Confusion matrix: \n ", cm)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation settings')

    parser = argparse.ArgumentParser(description='Evaluation settings')
    parser.add_argument('--data_root', nargs='+', default=['/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse19', 
                                                           '/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse20',
                                                           '/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-tri'])
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'])
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'])
    parser.add_argument('--master_list', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/Castellvi_list_Final_Split_v2.xlsx')
    parser.add_argument('--classification_type', default='right_side')
    parser.add_argument('--castellvi_classes', nargs='+', default=['1a', '1b', '2a', '2b', '3a', '3b', '4', '0'])
    parser.add_argument('--model', default='densenet')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--scheduler', default='ReduceLROnPlateau')
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--total_iterations', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_intervals', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--experiments', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments')
    parser.add_argument('--gpu_id', default='3')
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--port', type=int, default=6484)
    parser.add_argument('--dataset', nargs='+', default=['verse', 'tri'])
    parser.add_argument('--eval_type', type=str, default='test')
    parser.add_argument('--results_df', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/results.csv')

  
    parser.add_argument('--use_seg', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--weighted_sample', action='store_true')
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--flip_all', action='store_true')
    parser.add_argument('--version_no', type=int, default=0)
    parser.add_argument('--use_bin_seg', action='store_true')
    parser.add_argument('--use_zero_out', action='store_true')

    params = parser.parse_args()
    ckpt_path = params.experiments + '/baseline_models/' + params.model + '/best_models/version_' + str(params.version_no) 
    main(params=params, ckpt_path=ckpt_path)
