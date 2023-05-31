import sys
sys.path.append('/u/home/ank/3D-Castellvi-Prediction/bids')
import torch
import argparse
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
        masterlist = pd.read_excel("VerSe_masterlist_V3.xlsx", index_col=0)
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
            if records[index].subject in test_subjects:
                test_records.append(records[index])

        return test_records
    
    def get_predictions(self, model):

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
        for record in test_records:
            
            # Add true label to y_true
            y_true.append(str(record["castellvi"]))
            
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
                pred_cls = self.get_label_map.get((pred_cls_1, pred_cls_2))

                if pred_cls is None:
                    raise ValueError("Invalid prediction pair: ", pred_cls_1, pred_cls_2)
                
                y_pred.append(pred_cls)

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


                
def main(params):
    evaluator = Eval(opt=params)
    ckpt_path = ''    
    model = evaluator.load_model(path=ckpt_path, params=params)
    y_true, y_pred = evaluator.get_predictions(model)
    f1 = evaluator.get_f1_score(y_true, y_pred)
    cm = evaluator.get_confusion_matrix(y_true, y_pred)

    print("F1 score: ", f1)
    print("Confusion matrix: ", cm)

    evaluator.plot_confusion_matrix(cm, classes=['0', '2a', '2b', '3a', '3b', '4'], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/settings.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    params = read_config(args.settings)
    main(params=params)
