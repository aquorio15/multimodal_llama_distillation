from typing import List, Union
import logging
import os
import sys
import joblib
import glob
import fairseq
import pydub
import soundfile as sf
from multiprocessing import Pool
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from einops import rearrange
import re
import numpy as np
from functools import partial
import torch.multiprocessing as mp
import torchaudio
import glob
import tqdm
import argparse
from torchaudio.functional import resample
#%%
logging.basicConfig(
    format="%(asctime)s | pi%(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger('generate_pseudo_language')

def exact_div(x, y):
    assert x % y == 0
    return x // y

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 5
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE 
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

class FeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, fp16=False, sampling_rate=16000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(DEVICE)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate
        
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")

    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        wav = pad_or_trim(wav)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().cuda()
            else:
                x = x.float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            feat_chunk, _ = self.model.extract_features(
                        source=x,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
        if len(feat_chunk) == 0:
            return torch.zeros(0, 0)
        return feat_chunk

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    def __init__(
        self, 
        ckpt_dir,
        layer=11, 
        max_chunk=1600000, 
        fp16=False, 
        sampling_rate=16000,
        ):

        super().__init__()

        ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_germanic_it3.pt")
        km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

        self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, sampling_rate)
        #self.apply_kmeans = ApplyKmeans(km_path)
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def __call__(self, path, merged=True):
        waveform = self.feature_reader.read_audio(path).to(DEVICE)
        
        feat = self.feature_reader.get_feats(waveform)
        #cluster_ids = self.apply_kmeans(feat).tolist()
        # dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        # merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        # unmerged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"
        # if merged:
        #     return merged_units
        # else:
        #     return unmerged_units
        
        return feat
        
        # return {"continuous":feat, "units":dup_cluster_list, "duration":duration_list, "unmerged_units":cluster_ids}
        

def get_files_from_path(path: str = ".", ext=None) -> list:
    result = []
    for subdir, dirs, files in os.walk(path):
        for fname in files:
            filepath = f"{subdir}{os.sep}{fname}"
            if ext == None:
                result.append(filepath)
            elif type(ext) == str and fname.lower().endswith(ext.lower()):
                result.append(filepath)
            elif type(ext) == list:
                for item in ext:
                    if fname.lower().endswith(item.lower()):
                        result.append(filepath)
    return result

if __name__ == '__main__':

    ckpt_dir = "/nfsshare/Amartya/NAACL"

    #path = get_files_from_path('/DATA/nfsshare/Amartya/NAACL/shrutilipi/LibriLight/large_new', '.wav')
    # with open('/nfsshare/Amartya/NAACL/shrutilipi/mls_english/train/spanish.txt', 'r') as h:
    #     paths = h.readlines()
    #save_path = '/nfsshare/Amartya/Winoground/features/Hubert/German/'
    #os.makedirs(save_path, exist_ok=True)
    paths = glob.glob('/nfsshare/Amartya/A_question_answering/Audio/*/*/*.wav')
    s2u = Speech2Unit(
        ckpt_dir=ckpt_dir
    )
    for path in tqdm.tqdm(paths):
        name = os.path.basename(path).replace('.wav', '.npy')
        dirname = os.path.dirname(path).replace('Audio', 'features/Hubert')
        os.makedirs(dirname, exist_ok=True)
        if not os.path.exists(f"{dirname}/{name}"):
            units = s2u(path)
            np.save(f"{dirname}/{name}", units.cpu().numpy())
            del units
            torch.cuda.empty_cache()
# %%
