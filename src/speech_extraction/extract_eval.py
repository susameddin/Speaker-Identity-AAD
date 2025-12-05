import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from IPython.display import Audio

from metrics.extraction import Evaluator

sys.path.append("/engram/naplab/shared/spk_id_aad")

class Extraction(sb.Brain):
    """SpeechBrain Brain class for speaker-conditioned extraction."""

    def compute_forward(self, mix, clue, stage=None):
        # Encode mixture
        mix_h = self.hparams.Encoder(mix)
        
        # Estimate mask using conditioning clue (speaker embedding / label)
        est_mask = self.hparams.MaskNet(mix_h, clue)  # (1, B, F, T)
        est_mask = est_mask.squeeze(0)
        est_tar_h = mix_h * est_mask  # (B, F, T)

        # Decode estimated target waveform
        est_tar = self.hparams.Decoder(est_tar_h)
        
        # Match original time dimension after conv1d in encoder
        T_origin = mix.size(1)
        T_ext = est_tar.size(1)
        
        if T_origin > T_ext:
            est_tar = F.pad(est_tar, (0, T_origin - T_ext))
        else:
            est_tar = est_tar[:, :T_origin]
            
        return est_tar
    

LABEL = 'xvect8'
SPK_DIM = 512
HPARAMS = '/engram/naplab/shared/spk_id_aad/hparams/frontend/extractio.yaml'
CKPT_PATH = f'/engram/naplab/shared/spk_id_aad/extraction/xvect8_lr2e-4_ep50_SNR'
TEST_MANIFEST = '/engram/naplab/shared/spk_id_aad/manifests/test.json'

# Build argument list for SpeechBrain parser
argv = [HPARAMS, '--output_folder', CKPT_PATH]
argv += ['--train_data', 'null', '--valid_data', 'null']
argv += ['--which_label', LABEL]
argv += ['--spk_dim', str(SPK_DIM)]
argv += ['--batch_size', '1']
argv += ['--predicted', 'true']
argv += ['--test_manifest', TEST_MANIFEST]

hparams_file, run_opts, overrides = sb.parse_arguments(argv)
with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)
    
# Load best checkpoint according to SNR
loaded = hparams['checkpointer'].recover_if_possible(min_key='-snr')
print('Loading ckpts from:')
print(loaded)
print('-' * 100)

# Initialize extraction system
extractor = Extraction(
    modules=hparams["modules"],
    opt_class=hparams["optimizer"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
)

test_data = hparams['test_data']

mixes = []
tars = []
est_tars = []
pred_labels = []
att_labels = []
unatt_labels = []

# Evaluator computes objective and ASR-based metrics
evaluator = Evaluator(
    metrics=['SNR', 'SISNR', 'PESQ', 'WER', 'SIM', 'PredACC', 'ExtACC'],
    asr_model="openai/whisper-large-v3"
)

# Run extraction and evaluation on test set
for i in tqdm(range(len(test_data)), total=len(test_data)):
    mix, tar, clue, batch = test_data.__getitem__(i, debug=True)
    info = batch[0]
    
    with torch.no_grad():
        est_tar = extractor.compute_forward(mix.cuda(), clue.cuda())
        evaluator({
            'est_tar': est_tar.cuda(),
            'tar': tar.cuda(),
            'mix': mix.cuda(),
            'pred_label': info['speeches'][0]['pred_' + LABEL],
            'label_att': info['speeches'][0][LABEL],
            'label_unatt': info['speeches'][1][LABEL],
            'words': info['speeches'][0]['content_prompt'],
        })
        
    mixes.append(mix)
    est_tars.append(est_tar.cpu())
    tars.append(tar)
    pred_labels.append(info['speeches'][0]['pred_' + LABEL])
    att_labels.append(info['speeches'][0][LABEL])
    unatt_labels.append(info['speeches'][1][LABEL])
    
# Print averaged metrics
evaluator.summarize()