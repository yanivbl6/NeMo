# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from numpy import ndarray
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pesq import pesq
from pystoi import stoi
from torch import Tensor, nn

from nemo.collections.tts.helpers.helpers import griffin_lim
from nemo.collections.tts.models.base import MelToSpec
from nemo.collections.tts.modules.ed_mel2spec import OperationMode
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import IntType, LengthsType, SpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


@dataclass
class EDMel2SpecConfig:
    mel2spec: Dict[Any, Any] = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None
    train_params: Optional[Dict[Any, Any]] = None
    sched: Optional[Dict[Any, Any]] = None


def gen_filter(k):
    K = torch.ones(1, 1, k, 1)
    K.requires_grad = False
    return K


def EvalModule(clean, denoised, fs):
    """
    Evaluate the quality of the output audio in comparison with the real audio.
    Args:
        clean: the real audio
        denoised: the output audio
        fs: sampling rate. For PESQ, we use 16,000 as per the library limitation.
    Returns:
        STOI score computed with pystoi lib
        PESQ score computer with pesq lib
    """

    stoi_score = stoi(clean, denoised, fs, extended=False)
    pesq_score = pesq(16000, np.asarray(clean), denoised, 'wb')
    return stoi_score, pesq_score


def calc_using_eval_module(
    y_clean: ndarray, y_est: ndarray, T_ys: Sequence[int] = (0,), sampling_rate=22050
) -> Dict[str, float]:
    """
    calculate metric using EvalModule. y can be a batch.
    Args:
        y_clean: real audio
        y_est: estimated audio
        T_ys: length of the non-zero parts of the histograms
        sampling_rate: The used Sampling rate.

    Returns:
        A dictionary mapping scoring systems (string) to numerical scores.
    """

    if y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],) * y_clean.shape[0]

    stoi_result, pesq_results = EvalModule(y_clean[0, : T_ys[0]], y_est[0, : T_ys[0]], sampling_rate)

    return {'STOI': stoi_result, 'PESQ': pesq_results}


class EDMel2SpecModel(MelToSpec):
    """A model that convert mel spectrograms to linear spectrograms"""

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(EDMel2SpecConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        OmegaConf.merge(cfg, schema)

        self.ed_mel2spec = instantiate(self._cfg.mel2spec)
        self.mode = OperationMode.infer

        self.criterion = nn.L1Loss(reduction='none')  # maybe should be loss subclass?
        loss_mode = self._cfg.train_params.loss_mode
        self.lreg_factor = self._cfg.train_params.lreg_factor

        self.f_specs = {
            0: [(5, 2), (15, 5)],
            1: [(5, 2)],
            2: [(3, 1)],
            3: [(3, 1), (5, 2)],
            4: [(3, 1), (5, 2), (7, 3)],
            5: [(15, 5)],
            6: [(3, 1), (5, 2), (7, 3), (15, 5), (25, 10)],
            7: [(1, 1)],
            8: [(1, 1), (3, 1), (5, 2), (15, 5), (7, 3), (25, 10), (9, 4), (20, 5), (5, 3)],
            9: [(6, 2), (10, 4)],
        }[loss_mode]

        self.filters = [gen_filter(k) for k, s in self.f_specs]

        # self.l_hop = self._cfg.ed_mel2spec.hop_length
        # self.n_fft = self._cfg.ed_mel2spec.n_fft
        # self.kwargs_stft = dict(hop_length=self.l_hop, window='hann', center=True, n_fft=self.n_fft, dtype=np.float32)
        # self.kwargs_istft = dict(hop_length=self.l_hop, window='hann', center=True, dtype=np.float32)

    @property
    def input_types(self):
        return {
            "mel": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "spec": NeuralType(('B', 'C', 'D', 'T'), SpectrogramType()),
        }

    @typecheck()
    def forward(self, *, mel):
        if self.mode != self.ed_mel2spec.mode:
            raise ValueError(
                f"Encoder-Decoder Mel-to-Spec's mode {self.mode} does not match EDMel2SpecModule's mode {self.ed_mel2spec.mode}"
            )

        spec = self.ed_mel2spec(mel=mel)
        return spec  # audio_pred

    def convert_mel_spectrogram_to_linear(self, mel: torch.Tensor) -> torch.Tensor:
        self.eval()
        self.mode = OperationMode.infer
        self.ed_mel2spec.mode = OperationMode.infer

        return None

    def calc_loss(self, x: Tensor, y: Tensor, T_ys: Sequence[int], crit) -> Tensor:
        """
        x: B, C, F, T
        y: B, C, F, T
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot = 0
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        loss = loss_blocks.squeeze()

        return loss

    def calc_loss_smooth(
        self, _x: Tensor, _y: Tensor, T_ys: Sequence[int], kern: int, stride: int, pad: int = 0
    ) -> Tensor:
        """
        out_blocks: B, depth, C, F, T
        y: B, C, F, T
        """

        crit = self.criterion

        x = F.max_pool2d(_x, (kern, 1), stride=stride)
        y = F.max_pool2d(_y, (kern, 1), stride=stride)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot = 0
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        loss1 = loss_blocks.squeeze()

        x = F.max_pool2d(-1 * _x, (kern, 1), stride=stride)
        y = F.max_pool2d(-1 * _y, (kern, 1), stride=stride)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss_no_red = crit(x, y)

        loss_blocks = torch.zeros(x.shape[1], device=y.device)

        tot = 0
        for T, loss_batch in zip(T_ys, loss_no_red):
            tot += T
            loss_blocks += torch.sum(loss_batch[..., :T])
        loss_blocks = loss_blocks / tot

        loss2 = loss_blocks.squeeze()

        loss = loss1 + loss2
        return loss

    def training_step(self, batch, batch_idx):
        self.mode = OperationMode.training
        self.ed_mel2spec.mode = OperationMode.training
        _, y_spec, _, _, T_ys, _, _ = batch

        x_mel = self.ed_mel2spec.spec_to_mel(y_spec)

        x_spec = self(mel=x_mel)
        z_mel = self.ed_mel2spec.spec_to_mel(x_spec)

        loss_L1 = self.calc_loss(x_spec, y_spec, T_ys, self.criterion)
        loss_reg = self.calc_loss(x_mel, z_mel, T_ys, self.criterion)

        loss = loss_L1 + self.lreg_factor * loss_reg

        for (k, s) in self.f_specs:
            loss = loss + self.calc_loss_smooth(x_spec, y_spec, T_ys, k, s)

        output = {
            'loss': loss,
            'progress_bar': {'training_loss': loss},
            'log': {'loss': loss},
        }
        return output

    def validation_step(self, batch, batch_idx):

        self.mode = OperationMode.validation
        self.ed_mel2spec.mode = OperationMode.validation
        _, y_spec, _, _, T_ys, _, path_speech = batch

        x_mel = self.ed_mel2spec.spec_to_mel(y_spec)

        x_spec = self(mel=x_mel)
        z_mel = self.ed_mel2spec.spec_to_mel(x_spec)

        loss_L1 = self.calc_loss(x_spec, y_spec, T_ys, self.criterion)
        loss_reg = self.calc_loss(x_mel, z_mel, T_ys, self.criterion)

        loss = loss_L1 + self.lreg_factor * loss_reg

        output = {
            'val_loss': loss,
            'loss_L1': loss_L1,
            'loss_reg': loss_reg,
        }

        if self._cfg.train_params.validate_scores:
            '''
                For validaiton, estimate the wave using standard griffin lim, 
                comparing the real wave with the griffin lim counterpart.
            '''

            cnt = x_spec.shape[0]
            np_x = x_spec.to('cpu').numpy()
            np_y = y_spec.to('cpu').numpy()
            stoi_real, pesq_real, stoi_est, pesq_est = (0.0, 0.0, 0.0, 0.0)

            for p in range(cnt):
                y_wav_path = path_speech[p]
                wav = sf.read(y_wav_path)[0].astype(np.float32)

                y_est_wav = griffin_lim(np_y[p, 0, :, :])
                x_est_wav = griffin_lim(np_x[p, 0, :, :])

                min_size = min(wav.shape[0], x_est_wav.shape[0], y_est_wav.shape[0])
                wav = wav[0:min_size, ...]
                y_est_wav = y_est_wav[0:min_size, ...]
                x_est_wav = x_est_wav[0:min_size, ...]

                measure = calc_using_eval_module(x_est_wav, wav)
                stoi_real += torch.tensor(measure['STOI'])
                pesq_real += torch.tensor(measure['PESQ'])

                measure = calc_using_eval_module(x_est_wav, y_est_wav)
                stoi_est += torch.tensor(measure['STOI'])
                pesq_est += torch.tensor(measure['PESQ'])

            output['stoi_real'] = stoi_real / cnt
            output['pesq_real'] = pesq_real / cnt
            output['stoi_est'] = stoi_est / cnt
            output['pesq_est'] = pesq_est / cnt

        for (k, s) in self.f_specs:
            new_loss = self.calc_loss_smooth(x_spec, y_spec, T_ys, k, s)
            output[f'loss_{k}_{s}'] = new_loss
            loss = loss + new_loss

        output['val_loss'] = loss

        return output

    def validation_epoch_end(self, outputs):
        tensorboard_logs = {}

        for k in outputs[0].keys():
            tensorboard_logs[k] = torch.stack([x[k] for x in outputs]).mean()

        return {'val_loss': tensorboard_logs['val_loss'], 'log': tensorboard_logs}

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")  # TODO
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")  # TODO
        if shuffle_should_be:
            if 'shuffle' not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg["dataloader_params"]):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        dataset = instantiate(cfg.dataset)
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []

        # model = PretrainedModelInfo(
        #     pretrained_model_name="DeepGriffinLim-fft_1024-22050Hz",
        #     location="https://nemo-public.s3.us-east-2.amazonaws.com/nemo-1.0.0alpha-tests/DeepGriffinLim-fft_1024.nemo",  ##FIXME
        #     description="The model is trained on LJSpeech sampled at 22050Hz, n_fft=1024, with 6 layers and a widening factor of 16. Can be used as a trained alternative for GriffinLim",
        # )
        # list_of_models.append(model)
        return list_of_models
