import sys
sys.path.append('/data1/practical-sose23/castellvi/castellvi_prediction/bids')
import torch
import os
import argparse
import json
import re
from utils._prepare_data import DataHandler, read_config
from utils._get_model import *
from modules.DenseNetModule import DenseNet
from dataset.VerSe import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Eval:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.data_size = (128,86,136)
        self.num_classes = opt.num_classes
        self.softmax = nn.Softmax(dim=1)

    def _get_version_number(self, ckpt_path):
        version = re.search(r'version_(\d+)', ckpt_path)
        if version is None:
            raise ValueError("Invalid checkpoint path: cannot extract version number.")
        version_number = version.group(1)
        return version_number

    def get_best_model_paths(self, dir_path):
        best_model_paths = []
        for root, dirs, files in os.walk(dir_path):
            if files:  # if there are files in the directory
                # Extract validation loss from file name and choose the file with the smallest loss
                best_model_file = min(files, key=lambda f: float(re.search(r'val_loss=(\d+\.\d+)', f).group(1)))
                best_model_paths.append(os.path.join(root, best_model_file))
        return best_model_paths


    def get_label_map(self):
        if self.opt.classification_type == "right_side":
            # Define a mapping from pairs of predictions to final predictions
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
        else:
            raise NotImplementedError("Only right_side classification is supported")
                    
    def load_model(self, path, params):
        checkpoint = torch.load(path)
        model = DenseNet(opt=params, num_classes=self.num_classes, data_size=self.data_size, data_channel=1)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def get_test_dataset(self):
        # extract the test subjects from masterlist by usıng the Split column and if Flip is 0
        masterlist = pd.read_excel(self.opt.master_list, index_col=0)
        masterlist = masterlist.loc[masterlist['Split'] == 'test'] # extract test subjects
        masterlist = masterlist.loc[masterlist['Flip'] == 0] # extract subjects with Flip = 0
        test_subjects = masterlist.index.tolist()
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
        records = processor.verse_records
        # return speicific indexes of records that are in test_subjects
        for index in range(len(records)):
            if index in test_subjects:
                test_records.append(records[index])

        return test_records
    
    def get_predictions(self, model, version_number):
        # TODO : save each pred results for images
        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Set model to evaluation mode
        model.eval()

        # Get processor obejct
        processor = self.get_processor()

        # Get test subjects
        test_subjects = self.get_test_dataset()

        # Get records
        test_records = self.get_records(processor, test_subjects)

        y_pred = []
        y_true = []
        eval_results = []

        # Check if eval_results file exists and load it
        eval_file_path = self.opt.experiments + "/eval_results.json"
        if os.path.exists(eval_file_path):
            with open(eval_file_path, 'r') as f:
                eval_results = json.load(f)

        # Convert eval_results into a dictionary for easy update
        eval_results_dict = {item['subject_name']: item for item in eval_results}

        for record in test_records:
            
            # Get the data
            img = processor._get_cutout(record,return_seg=self.opt.use_seg, max_shape=self.data_size)
            flip_img = np.flip(img, axis=2).copy() # Flip the image along the z-axis. In other words, flip the image horizontally
        
            # Add new axis to the image
            img = img[np.newaxis,np.newaxis, ...]
            flip_img = flip_img[np.newaxis, np.newaxis, ...]

            # Convert to tensor
            img = torch.from_numpy(img)
            flip_img = torch.from_numpy(flip_img)

            # Convert to float
            img = img.float()
            flip_img = flip_img.float()

            # Move to GPU
            img = img.to(device)
            flip_img = flip_img.to(device)

            # Get the label
            label = str(record["castellvi"]) # 0, 2a, 2b, 3a, 3b, 4
            y_true.append(label)

            with torch.no_grad():
                output_1 = model(img)
                output_2 = model(flip_img)

                pred_1 = self.softmax(output_1)
                pred_2 = self.softmax(output_2)

                _, pred_cls_1 = torch.max(pred_1, 1)
                _, pred_cls_2 = torch.max(pred_2, 1)

                pred_cls_1 = str(pred_cls_1.item())
                pred_cls_2 = str(pred_cls_2.item())

                # Determine the final prediction based on the pair of predictions
                pred_cls = self.get_label_map().get((pred_cls_1, pred_cls_2))

                if pred_cls is None:
                    raise ValueError("Invalid prediction pair: ", pred_cls_1, pred_cls_2)
                
                y_pred.append(pred_cls)

            
             # Update eval_results_dict
            subject_results = eval_results_dict.get(record['subject'], {
                'subject_name': record['subject'],
                'actual_label': label,
                'predicted_labels': {}
            })
            subject_results['predicted_labels'][f'experiment_{version_number}'] = pred_cls
            if pred_cls == label:
                subject_results['correct'] = subject_results.get('correct', 0) + 1
            else:
                subject_results['incorrect'] = subject_results.get('incorrect', 0) + 1
            eval_results_dict[record['subject']] = subject_results

        # Convert eval_results_dict back into a list
        eval_results = list(eval_results_dict.values())

        with open(eval_file_path, 'w') as f:
            json.dump(eval_results, f, indent=4)

        return y_true, y_pred
        
    def get_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        return f1
    
    def get_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        # save confusion matrix as plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=cmap)
        plt.savefig('confusion_matrix.png')
        plt.close()


                
def main(params, ckpt_path=None):
    evaluator = Eval(opt=params)
    # Get paths to the best model
    best_model= os.listdir(ckpt_path)[0]
    best_model_path = [os.path.join(ckpt_path, best_model)]


    # Run evaluation for each best model

    model = evaluator.load_model(path=best_model_path, params=params)

    y_true, y_pred = evaluator.get_predictions(model, version_number=params.version_no)
    f1 = evaluator.get_f1_score(y_true, y_pred)
    cm = evaluator.get_confusion_matrix(y_true, y_pred)

    print('Version Number:', params.version_no)
    print("Model: ", ckpt_path)
    print("F1 score: ", f1)
    print("Confusion matrix: \n ", cm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--data_root', nargs='+', default=['/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse19', '/data1/practical-sose23/castellvi/3D-Castellvi-Prediction/data/dataset-verse20'])
    parser.add_argument('--data_types', nargs='+', default=['rawdata', 'derivatives'])
    parser.add_argument('--img_types', nargs='+', default=['ct', 'subreg', 'cortex'])
    parser.add_argument('--master_list', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/src/dataset/VerSe_masterlist_V4.xlsx')
    parser.add_argument('--classification_type', default='right_side')
    parser.add_argument('--model', default='densenet')
    parser.add_argument('--use_seg', type=bool, default=False)
    parser.add_argument('--experiments', default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--version_no', type=int, default=0)


    params = parser.parse_args()
    ckpt_path = params.experiments + '/' + params.model + '/best_models/' + params.versiion_no 
    main(params=params, ckpt_path=ckpt_path)

