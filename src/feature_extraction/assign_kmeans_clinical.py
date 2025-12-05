import os
import json
import torch
import librosa

from copy import deepcopy
from speechbrain.inference.speaker import EncoderClassifier

def compute_spkemb(path):
    """Compute x-vector embedding for a wav file."""
    wav, sr = librosa.load(path, sr=16000)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).cuda()  # [1, T]
    with torch.no_grad():
        emb = xmodel.encode_batch(wav).squeeze()  # [emb_dim]
    return emb

clip_root = '/engram/naplab/shared/spk_id_aad/Clips'
manifest_file = os.path.join(clip_root, 'labels.json')
out_file = os.path.join(clip_root, 'labels.json')

with open(manifest_file, 'r') as f:
    all_clips = json.load(f)

# Load x-vector model
xmodel = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="xvect/spkrec-xvect-voxceleb",
    run_opts={"device": "cuda"}
)

# Paths to k-means centroids for different K
cluster_paths = {
    'xvect2': '/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K=2.pt',
    'xvect4': '/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K=4.pt',
    'xvect8': '/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K=8.pt',
    'xvect16': '/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K=16.pt',
    'xvect32': '/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K=32.pt',
    'xvect64': '/engram/naplab/shared/spk_id_aad/xvect/kmeans_utt10k/K=64.pt',
}

# Load all centroid tensors on GPU
clusters = {
    name: torch.load(path).cuda()
    for name, path in cluster_paths.items()
}

# Work on a copy of the manifest
labeled_clips = deepcopy(all_clips)

for trial, trial_clips in labeled_clips.items():
    print(f'Processing {str(trial)}...')
    for c, clip in enumerate(trial_clips):
        # Normalize relative paths to use {root} placeholder
        clip['Conv1Path'] = clip['Conv1Path'].replace('./', '{root}/')
        clip['Conv2Path'] = clip['Conv2Path'].replace('./', '{root}/')
        clip['BGPath'] = clip['BGPath'].replace('./', '{root}/')

        # Paths for saving x-vector tensors
        clip['XVector1Path'] = clip['Conv1Path'].replace('.wav', '_xvect.pt')
        clip['XVector2Path'] = clip['Conv2Path'].replace('.wav', '_xvect.pt')

        # Process both speakers in the mixture
        for spk in ['1', '2']:
            wav_path = clip[f'Conv{spk}Path'].format(root=clip_root)
            xvect = compute_spkemb(wav_path)

            # Save embedding tensor and also store as a list in JSON if needed
            torch.save(xvect, clip[f'XVector{spk}Path'].format(root=clip_root))
            clip[f'_xvect{spk}'] = xvect.tolist()

            # Assign cluster labels for each K
            for name, centroids in clusters.items():
                dists = torch.norm(centroids - xvect, dim=1)
                label = int(torch.argmin(dists))
                clip[f'Conv{spk}Labels'][name] = label

        # Move 'alignment' to the end of the dict (for readability)
        sort_clip = {key: clip[key] for key in sorted(clip.keys()) if key != 'alignment'}
        sort_clip['alignment'] = clip['alignment']
        trial_clips[c] = sort_clip

with open(out_file, 'w') as f:
    json.dump(labeled_clips, f, indent=4)