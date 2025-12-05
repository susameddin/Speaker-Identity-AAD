import csv
import torch
import librosa
import numpy as np
from tqdm import tqdm

def load_csv_to_dict(filename):
    """Load CSV manifest as a list of dicts."""
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data

def group_by_speakers(utterances):
    """Group utterances by speaker ID."""
    spks = {}
    for utt in utterances:
        spk = utt['spk']
        if spk not in spks:
            spks[spk] = [utt]
        else:
            spks[spk].append(utt)
    return spks

src_manifest_file = '/engram/naplab/shared/spk_id_aad/manifests/LibriTTS_longer1p5s.csv'
utterances = load_csv_to_dict(src_manifest_file)
utt_by_spk = group_by_speakers(utterances)

print('Number of utterances:', len(utterances))
print('Number of speakers:', len(utt_by_spk))
print(utterances[0].keys())

# Number of clusters and corresponding centroid file
K = 8
clusters_path = f'/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K={str(K)}.pt'
clusters = torch.load(clusters_path).cuda()
print(clusters.shape)

# Precomputed x-vector embeddings for all utterances
embs_path = '/engram/naplab/shared/spk_id_aad/xvect/all_embs_1p5s.pt'
embs = torch.load(embs_path).cuda()
print(embs.shape)

outs = []
label_count = [0 for _ in range(K)]

# Assign each utterance to nearest cluster in x-vector space
for utt, emb in tqdm(zip(utterances, embs), total=len(utterances)):
    dists = torch.norm(clusters - emb, dim=1)
    label = int(torch.argmin(dists))
    utt[f'xvect{str(K)}'] = label
    outs.append(utt)
    label_count[label] += 1

label_count = np.array(label_count)
label_perc = label_count / sum(label_count)

# Overwrite manifest with new cluster labels added
with open(src_manifest_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=outs[0].keys())
    writer.writeheader()
    writer.writerows(outs)