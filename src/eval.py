import os
from dataset.VerSe import VerSe
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix
from modules.DenseNetModule import DenseNet
from utils._prepare_data import DataHandler
from utils.environment_settings import env_settings
from tqdm import tqdm
import sys
import torch

class Eval:
    def __init__(self, params):
        self.params = params
        self.master_list = env_settings.MASTER_LIST
        self.dataset = [os.path.join(env_settings.DATA, "dataset-verse19"), 
                        os.path.join(env_settings.DATA, "dataset-verse20"), 
                        os.path.join(env_settings.DATA, "dataset-tri")]
        self.data_types = ['rawdata',"derivatives"]
        self.image_types = ["ct", "subreg", "cortex"]
        self.model_path = params.model_path

    def load_model(self, model_path):
        model = DenseNet.load_from_checkpoint(model_path)
        return model
    
    def get_processor(self):
        processor = DataHandler(master_list=self.master_list, dataset=self.dataset, data_types=self.data_types, image_types=self.image_types)
        return processor
    
    def get_verse_dataset(self, processor, model):
        verse_dataset = VerSe(model.opt, processor, processor.verse_records + processor.tri_records, training=False)
        return verse_dataset
    
    def get_val_subjects(self, verse_dataset):
        # For each record with dataset_split = val, get index of both flipped and non-flipped records
        val_subjects = [record["subject"] for record in verse_dataset.records if (record["dataset_split"] == "val" and record["flip"] == 1)]
        print('length of val dataset:', len(val_subjects))
        return val_subjects

    
    def get_joined_subjects(self, verse_dataset, val_subjects):
        val_subs_joined = {}
        val_subs_idx = []
        for subject in val_subjects:
            val_subs_joined[subject] = {}

        for index, record in enumerate(verse_dataset.records):
            if record["subject"] in val_subjects:
                val_subs_idx.append(index)
                if record["flip"] == 1:
                    val_subs_joined[record["subject"]]["flip"] = index
                    val_subs_joined[record["subject"]]["castellvi"] = record["castellvi"]
                    val_subs_joined[record["subject"]]["side"] = record["side"]
                else:
                    val_subs_joined[record["subject"]]["non_flip"] = index


        return val_subs_joined


    def full_castellvi_to_lbl(self, record):
        if record["castellvi"] in ("0", "1a", "1b"):
            return 0
        
        elif record["castellvi"] == "2a":
            if record["side"] == "R":
                return 1
            else:
                return 2
        elif record["castellvi"] == "2b":
            return 3
        elif record["castellvi"] == "3a":
            if record["side"] == "R":
                return 4
            else:
                return 5
        elif record["castellvi"] == "3b":
            return 6
        elif record["castellvi"] == "4":
            if record["side"] == "R":
                return 7
            else:
                return 8
        else:
            raise ValueError(f"Invalid castellvi value {record['castellvi']}")
        
    def get_no_side_castellvi_without_side(self, record):
        if record["castellvi"] in ["0", "1a", "1b"]:
            return 0
        elif record["castellvi"] == "2a":
            return 1
        elif record["castellvi"] == "2b":
            return 2
        elif record["castellvi"] == "3a":
            return 3
        elif record["castellvi"] == "3b":
            return 4
        elif record["castellvi"] == "4":
            return 5
    
    def get_no_side_castellvi_from_sides(self, y_flip, y_non_flip):
        if y_non_flip == 0 and y_flip == 0:
            #this is a 0 case
            return 0
        elif (y_non_flip == 0 and y_flip == 1) or (y_non_flip == 1 and y_flip == 0):
            #this is a 2a case
            return 1
        elif (y_non_flip == 1 and y_flip == 1):
            #this is a 2b case
            return 2
        elif (y_non_flip == 0 and y_flip == 2) or (y_non_flip == 2 and y_flip == 0):
            #this is a 3a case
            return 3
        elif y_non_flip == 2 and y_flip == 2:
            #this is a 3b case
            return 4
        elif (y_non_flip == 1 and y_flip == 2) or (y_non_flip == 2 and y_flip == 1):
            #this is a 4 case
            return 5


    def get_castellvi_from_sides(self, y_flip, y_non_flip):
        if y_non_flip == 0:
            if y_flip == 0:
                #This is a 0 casse
                return 0
            elif y_flip == 1:
                #This is a left-sided 2a casse
                return 2
            elif y_flip == 2:
                #This is a left-sided 3a casse
                return 5
        elif y_non_flip == 1:
            if y_flip == 0:
                #This is a right-sided 2a casse
                return 1
            elif y_flip == 1:
                #This is a 2b casse
                return 3
            elif y_flip == 2:
                #This is a right-sided 4 case
                return 7
        elif y_non_flip == 2:
            if y_flip == 0:
                #This is a right-sided 3a casse
                return 4
            elif y_flip == 1:
                #This is a 3b casse
                return 6
            elif y_flip == 2:
                #This is a left-sided 4 case
                return 8


    def evaluate(self):
        model = self.load_model(self.model_path)
        processor = self.get_processor()
        verse_dataset = self.get_verse_dataset(processor, model)
        val_subjects = self.get_val_subjects(verse_dataset)
        val_subs_joined = self.get_joined_subjects(verse_dataset, val_subjects)


        
        sidewise_pred = []
        sidewise_true = []
        full_castellvi_pred = []
        full_castellvi_true = []
        no_side_castellvi_pred = []
        no_side_castellvi_true = []
        failed_subs = []

        results = []

        classification_type = model.opt.classification_type
        zero_out = model.opt.use_zero_out
        dropout = model.opt.dropout_prob

        if model.opt.elastic_transform:
            augmentation = "Rand3DElastic"
        else:   
            augmentation = "RandAffine"


        if model.opt.use_seg:
            data_type = "seg"
        if model.opt.use_bin_seg:
            data_type = "bin_seg"
        if model.opt.use_seg_and_raw:
            data_type = "seg_and_ct"

        if not model.opt.use_seg_and_raw and not model.opt.use_bin_seg and not model.opt.use_seg:
            data_type = "ct"


        model.to('cuda')
        model.eval()
        for val_sub in tqdm(val_subs_joined):
            side_wise_pred = None
            side_wise_flip_pred = None
            side_wise_gt = None
            side_wise_flip_gt = None

            for flip in ["flip", "non_flip"]:
                idx = val_subs_joined[val_sub][flip]
                sidewise_true.append(verse_dataset[idx]["class"].to("cuda"))
                x = verse_dataset[idx]["target"]
                #convert x to cuda tensor
                x = x.unsqueeze(0)
                out = model(x.cuda())
                out = out.squeeze(0)
                sidewise_pred.append(out.argmax().item())

                if flip == "flip":
                    side_wise_flip_pred = out.argmax().item()
                    side_wise_gt = verse_dataset[idx]["class"].cpu()
                else:
                    side_wise_pred = out.argmax().item()
                    side_wise_flip_gt = verse_dataset[idx]["class"].cpu()

                if out.argmax().item() != verse_dataset[idx]["class"].cpu():
                    failed_subs.append({"subject": val_sub, "side": "L" if flip == "flip" else "R", "y_true": verse_dataset[idx]["class"].cpu(), "y_pred": out.argmax().item(), "castellvi": val_subs_joined[val_sub]["castellvi"]})
                if flip == "flip":
                    y_pred_flip = out.argmax().item()
                else:
                    y_pred_non_flip = out.argmax().item()

            # Combine y_pred_flip and y_pred_non_flip to final Castellvi prediction
            full_castellvi_pred.append(self.get_castellvi_from_sides(y_pred_flip, y_pred_non_flip))
            full_castellvi_true.append(self.full_castellvi_to_lbl(val_subs_joined[val_sub]))

            # Add Side-Agnostic Castellvi Label
            no_side_castellvi_pred.append(self.get_no_side_castellvi_from_sides(y_pred_flip, y_pred_non_flip))
            no_side_castellvi_true.append(self.get_no_side_castellvi_without_side(val_subs_joined[val_sub]))


            pred  = self.get_no_side_castellvi_from_sides(y_pred_flip, y_pred_non_flip)
            gt = self.get_no_side_castellvi_without_side(val_subs_joined[val_sub])

            
            no_side_map = {
                0 : "0",
                1 : "2a",
                2 : "2b",
                3 : "3a",
                4 : "3b",
                5 : "4"
            }

            side_wise_map = {
                0 : "0",
                1 : "2",
                2 : "3",
            }


            subject_result_dict = {
                "subject": val_sub,
                "experiment_no": self.params.version, 
                'gt': no_side_map[gt],
                'pred': no_side_map[pred],
                'side_wise_gt': side_wise_map[side_wise_gt.cpu().item()],
                'side_wise_pred': side_wise_map[side_wise_pred],
                'side_wise_flip_gt': side_wise_map[side_wise_flip_gt.cpu().item()],
                'side_wise_flip_pred': side_wise_map[side_wise_flip_pred],
            }

            results.append(subject_result_dict)


        import pandas as pd
        # create dataframe from results array first check if results.csv exists if it is append to it
        if os.path.exists("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/results.csv"):
            df = pd.read_csv("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/results.csv")
            df = df.append(pd.DataFrame(results))
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.to_csv("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/results.csv")
        else:
            df = pd.DataFrame(results)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.to_csv("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/results.csv")


        sidewise_true = [y.cpu() for y in sidewise_true]

        cm = confusion_matrix(sidewise_true, sidewise_pred)
        print(cm)
        # convert confusion matrix to list
        cm = cm.tolist()
        print(cm)


        side_wise_accuracy = accuracy_score(sidewise_true, sidewise_pred)
        side_wise_f1 = f1_score(sidewise_true, sidewise_pred, average="weighted")
        side_wise_mcc = matthews_corrcoef(sidewise_true, sidewise_pred)
        side_wise_kappa = cohen_kappa_score(sidewise_true, sidewise_pred)


        accuracy = accuracy_score(no_side_castellvi_true, no_side_castellvi_pred)
        f1 = f1_score(no_side_castellvi_true, no_side_castellvi_pred, average="weighted")
        mcc = matthews_corrcoef(no_side_castellvi_true, no_side_castellvi_pred)
        kappa = cohen_kappa_score(no_side_castellvi_true, no_side_castellvi_pred)


        metrics = {
            'experiment_no': self.params.version,
            "cutout_type": classification_type,
            "data_type": data_type,
            "zero_out": zero_out,
            "augmentation" : augmentation,
            "dropout_prob": dropout,
            "side_wise_accuracy": side_wise_accuracy,
            "side_wise_f1": side_wise_f1,
            "side_wise_mcc": side_wise_mcc,
            "side_wise_kappa": side_wise_kappa,
            'side_wise_cm': cm,
            "accuracy": accuracy,
            "f1": f1,
            "mcc": mcc,
            "kappa": kappa,

        }

        # create dataframe from results array first check if metrics.csv exists if it is append to it
        if os.path.exists("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/metrics.csv"):
            df_metrics = pd.read_csv("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/metrics.csv")
            df_metrics = df_metrics.append(pd.DataFrame([metrics]))
            df_metrics = df_metrics.loc[:, ~df_metrics.columns.str.contains('^Unnamed')]
            df_metrics.to_csv("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/metrics.csv")

        else:
            df_metrics = pd.DataFrame([metrics])
            df_metrics = df_metrics.loc[:, ~df_metrics.columns.str.contains('^Unnamed')]
            df_metrics.to_csv("/data1/practical-sose23/castellvi/team_repo/3D-Castellvi-Prediction/experiments/baseline_models/densenet/metrics.csv")




if __name__ == '__main__':

    if env_settings.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(env_settings.CUDA_VISIBLE_DEVICES)

    if env_settings.BIDS_PATH is not None:
        sys.path.append(env_settings.BIDS_PATH)

    if torch.cuda.is_available():
        print('Running on GPU #' + str(torch.cuda.current_device()))
    else:
        print('Running on CPU')


    import argparse
    parser = argparse.ArgumentParser(description='Training settings')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--version', type=int, default=0)
    params = parser.parse_args()

    evaluator = Eval(params)
    evaluator.evaluate()



    
