import os
import random
import json
import torch
import scipy.io
import numpy as np
from scipy.stats import zscore
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class NeuroLabelData(Dataset):
    """
    Dataset for loading neural clips and corresponding labels.
    """
    def __init__(self, clip_root, label_file, label_type,
                 elec_select=False, elec_list=None,
                 norm_data=True, preds_save=False):
        self.root = clip_root
        with open(label_file, 'r') as f:
            self.clips = json.load(f)

        self.label_type = label_type
        self.elec_list = elec_list
        self.norm_data = norm_data
        self.preds_save = preds_save

        # Hard-coded speaker IDs for speaker-label setting
        if self.label_type == 'speaker':
            self.speaker_to_id = {
                'A': 0, 'D': 1, 'I': 2, 'J': 3,
                'C': 4, 'E': 5, 'F': 6, 'G': 7
            }

        # For style-based labels, drop clips without valid labels
        if self.label_type.startswith("style"):
            for clip in self.clips:
                if int(clip['Conv1Labels'][self.label_type]) == -10:
                    self.clips.remove(clip)
        
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]

        # Build neural data path
        neuro_path = os.path.join(
            "{root}/Clips",
            "NeuralData",
            clip['BGPath'].split("/")[-1][:-4] + "_resp_lf.pt"
        )
        if self.norm_data:
            neuro_path = neuro_path[:-3] + "_norm_all.pt"

        neuro = torch.load(neuro_path.format(root=self.root))

        # Electrode selection
        if self.elec_list is not None:
            neuro = neuro[:, self.elec_list]

        # Map label based on label type
        if self.label_type == 'speaker':
            label = self.speaker_to_id[clip['speaker']]
        else:
            label = int(clip['Conv1Labels'][self.label_type])
        
        # Optionally return IDs for saving predictions later
        if self.preds_save:
            return neuro, label, clip['trail_id'], clip['sentence_id']
        else:
            return neuro, label
    

class RNN(nn.Module):
    """
    Simple RNN classifier over neural time series.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers, bidirectional=True):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Normalize features per time step
        x = F.layer_norm(x, [self.input_dim])
        rnn_out, _ = self.rnn(x)
        last_rnn_out = torch.mean(rnn_out, 1)
        out = self.fc(last_rnn_out)
        return out


# Paths and channel selection
clip_root = '/engram/naplab/shared/spk_id_aad/Clips'
train_file = os.path.join(clip_root, 'labels_train.json')
valid_file = os.path.join(clip_root, 'labels_valid.json')
test_file = os.path.join(clip_root, 'labels_test.json')

# Load list of bad channels selected by visual inspection and obtain good ones
with open(os.path.join(clip_root, 'Data/badchannels_lowerlim.txt')) as file:
    bad_elecs = [int(line.rstrip()) - 1 for line in file]
good_elecs = np.setdiff1d(np.arange(1311), np.array(bad_elecs))

subject_id = 1
subject_list = np.load(os.path.join(clip_root, 'Data/subject_list.npy'))
elec_list = np.intersect1d(np.where(subject_list == subject_id), good_elecs)

# Config
CONFIGS = {
    'label_type': 'xvect8',
    'n_electrode': len(elec_list),
    'rnn_dim': 64,
    'n_layer': 1,
    'out_dim': 8,
    'bidirectional': True,
    'lr': 1e-4,
    'n_epoch': 100,
    'elec_list': elec_list,
    'norm_data': True,
    'save_predictions': False
}

# Datasets
train_data = NeuroLabelData(
    clip_root=clip_root,
    label_file=train_file,
    label_type=CONFIGS['label_type'],
    elec_list=CONFIGS['elec_list'],
    norm_data=CONFIGS['norm_data']
)

valid_data = NeuroLabelData(
    clip_root=clip_root,
    label_file=valid_file,
    label_type=CONFIGS['label_type'],
    elec_list=CONFIGS['elec_list'],
    norm_data=CONFIGS['norm_data']
)

test_data = NeuroLabelData(
    clip_root=clip_root,
    label_file=test_file,
    label_type=CONFIGS['label_type'],
    elec_list=CONFIGS['elec_list'],
    norm_data=CONFIGS['norm_data']
)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)

# Model, loss, optimizer
predictor = RNN(
    input_dim=CONFIGS['n_electrode'],
    hidden_dim=CONFIGS['rnn_dim'],
    output_dim=CONFIGS['out_dim'],
    num_layers=CONFIGS['n_layer'],
    bidirectional=CONFIGS['bidirectional']
).cuda()

loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(predictor.parameters(), lr=CONFIGS['lr'])

best_accuracy = 0
label_type_str = str(CONFIGS['label_type'])
save_path = os.path.join(
    clip_root,
    f'save_neuro/bf/subjects/{label_type_str}_subj{subject_id}.pt'
)

# Training + validation loop
for epoch in range(CONFIGS['n_epoch']):
    predictor.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, label in tqdm(train_loader, total=len(train_loader)):
        x = x.float().cuda()
        label = label.cuda()
        
        optimizer.zero_grad()
        y = predictor(x)
        loss = loss_fn(y, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, est_label = torch.max(y.data, 1)
        total += label.size(0)
        correct += (est_label == label).sum().item()

    accuracy = 100 * correct / total
    print(
        f"Epoch Train {epoch+1}/{CONFIGS['n_epoch']}, "
        f"Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}%"
    )

    # Validation
    predictor.eval()
    total_loss = 0
    correct = 0
    total = 0
    for x, label in tqdm(valid_loader, total=len(valid_loader)):
        x = x.float().cuda()
        label = label.cuda()
        with torch.no_grad():
            y = predictor(x)
            loss = loss_fn(y, label)
        total_loss += loss.item()
        _, est_label = torch.max(y.data, 1)
        total += label.size(0)
        correct += (est_label == label).sum().item()

    accuracy = 100 * correct / total

    print('Saving the latest model.')
    torch.save(predictor.state_dict(), save_path)
        
    print(
        f"Epoch Valid {epoch+1}/{CONFIGS['n_epoch']}, "
        f"Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}%"
    )


# Test evaluation
predictor.load_state_dict(torch.load(save_path))
predictor.eval()
total_loss = 0
correct = 0
total = 0

if CONFIGS['save_predictions']:
    preds = []

for batch in tqdm(test_loader, total=len(test_loader)):
    # batch: (x, label) or (x, label, trail_id, sentence_id)
    x, label = batch[0:2]
    x = x.float().cuda()
    label = label.cuda()
    with torch.no_grad():
        y = predictor(x)
        loss = loss_fn(y, label)
    total_loss += loss.item()
    _, est_label = torch.max(y.data, 1)
    total += label.size(0)
    correct += (est_label == label).sum().item()
    
    if CONFIGS['save_predictions']:
        preds.append((batch[2], batch[3], est_label, label))
    
accuracy = 100 * correct / total
print(f"Test:, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}%")

# Save predictions back into JSON (optional)
if CONFIGS['save_predictions']:
    test_pred_file = os.path.join(clip_root, 'labels_test_preds.json')
    
    with open(test_pred_file, 'r') as f:
        clips_test = json.load(f)

    label_type = CONFIGS['label_type']
    
    for clip in clips_test:
        curr_label = [
            int(item[3])
            for item in preds
            if item[0] == clip['trail_id'] and item[1] == clip['sentence_id']
        ]
        curr_pred = [
            int(item[2])
            for item in preds
            if item[0] == clip['trail_id'] and item[1] == clip['sentence_id']
        ]

        if len(curr_pred) == 1 and len(curr_label) == 1:
            # Ground-truth label
            label_dict = clip['Conv1Labels']
            label_dict[label_type] = curr_label[0]
            clip['Conv1Labels'] = label_dict

            # Predicted label
            if 'PredConv1Labels' not in clip:
                clip['PredConv1Labels'] = {}
            
            pred_dict = clip['PredConv1Labels']
            pred_dict[label_type] = curr_pred[0]
            clip['PredConv1Labels'] = pred_dict

        else:
            print("No Predictions!")
        
    with open(test_pred_file, 'w') as f:
        json.dump(clips_test, f, indent=4)