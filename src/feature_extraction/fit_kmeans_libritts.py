import os
import csv
import torch
import librosa
from kmeans_pytorch import kmeans
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

def load_csv_to_dict(filename):
    """Load CSV manifest as a list of dicts."""
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return list(reader)

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

def compute_spkemb(path):
    """Compute x-vector embedding for a wav file."""
    wav, sr = librosa.load(path, sr=16000)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).cuda()
    emb = xmodel.encode_batch(wav).squeeze().detach().cpu()
    return emb

src_manifest_file = '/engram/naplab/shared/spk_id_aad/manifests/LibriTTS_longer1p5s.csv'
utterances = load_csv_to_dict(src_manifest_file)
utt_by_spk = group_by_speakers(utterances)

print('Number of utterances:', len(utterances))
print('Number of speakers:', len(utt_by_spk))

# Load x-vector model
xmodel = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="xvect/spkrec-xvect-voxceleb",
    run_opts={"device": "cuda"}
)

all_embs = []

# Compute embeddings for all utterances
for utt in tqdm(utterances, total=len(utterances)):
    wav_path = utt['raw_path']
    emb = compute_spkemb(wav_path)
    all_embs.append(emb)

all_embs = torch.stack(all_embs)

# Subsample for k-means (memory-friendly)
S = 10_000
sub_embs = all_embs[torch.randperm(all_embs.shape[0])[:S]]

cluster_save_root = f'/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt{str(int(S/1000))}k'
os.makedirs(cluster_save_root, exist_ok=True)

# Run k-means for several K values
label2, cluster2 = kmeans(sub_embs, num_clusters=2, distance='euclidean', device=torch.device('cuda'))
torch.save(cluster2, os.path.join(cluster_save_root, 'K=2.pt'))

label4, cluster4 = kmeans(sub_embs, num_clusters=4, distance='euclidean', device=torch.device('cuda'))
torch.save(cluster4, os.path.join(cluster_save_root, 'K=4.pt'))

label8, cluster8 = kmeans(sub_embs, num_clusters=8, distance='euclidean', device=torch.device('cuda'))
torch.save(cluster8, os.path.join(cluster_save_root, 'K=8.pt'))

label16, cluster16 = kmeans(sub_embs, num_clusters=16, distance='euclidean', device=torch.device('cuda'))
torch.save(cluster16, os.path.join(cluster_save_root, 'K=16.pt'))

label32, cluster32 = kmeans(sub_embs, num_clusters=32, distance='euclidean', device=torch.device('cuda'))
torch.save(cluster32, os.path.join(cluster_save_root, 'K=32.pt'))

label64, cluster64 = kmeans(sub_embs, num_clusters=64, distance='euclidean', device=torch.device('cuda'))
torch.save(cluster64, os.path.join(cluster_save_root, 'K=64.pt'))