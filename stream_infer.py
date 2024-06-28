#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""HiFiGAN streaming inference."""


import os
import argparse
import logging
import time
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

import torch
from typing import List
import math

import onnxruntime
import json
from json import JSONEncoder
import h5py

from parallel_wavegan.datasets import MelDataset


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Model:
    def __init__(
        self,
        model_path,
        providers: List[str],
    ):
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2
        self.model = onnxruntime.InferenceSession(
            model_path, sess_options, providers=providers
        )
        self.input_names = [d.name for d in self.model.get_inputs()]
        self.input_shapes = [d.shape for d in self.model.get_inputs()]
        self.output_names = [d.name for d in self.model.get_outputs()]
        logging.info(f"input names {self.input_names}")
        logging.info(f"output names {self.output_names}")
        self.reset_input_output_data()

    def __call__(self, input_dict: dict):
        out = self.model.run(self.output_names, input_dict)
        out = {key: value for key, value in zip(self.output_names, out)}
        self.input_output_data.append({"input": input_dict, "output": out})
        return out

    def get_input_names(self):
        return self.input_names

    def get_output_names(self):
        return self.output_names

    def get_input_output_data(self):
        return self.input_output_data

    def dump_to_json(self, name):
        infer_data = self.get_input_output_data()
        with open(name, "w") as f:
            json.dump(infer_data, f, cls=NumpyArrayEncoder)

    def reset_input_output_data(self):
        self.input_output_data = []  # format: (input, output)


def check_format_in_directory(directory):
    has_h5 = False
    has_npy = False

    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            has_h5 = True
        elif filename.endswith(".npy"):
            has_npy = True

    if has_h5 and not has_npy:
        return "hdf5"
    elif has_npy and not has_h5:
        return "npy"
    else:
        return None


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def stream_inference(model, batch, chunk_size, cumulative_delays):
    input_shapes = model.input_shapes
    cache_size = input_shapes[1][1]
    logging.info(f"cache size {cache_size}")
    input_cache = np.zeros([1, cache_size], dtype=np.float32)
    xs = batch["mel"].transpose(0, 1).unsqueeze(0).cpu().numpy()
    real_mel_length = xs.shape[2]
    outputs = []
    cur = 0
    while cur < xs.shape[2]:
        end = min(cur + chunk_size, xs.shape[2])
        chunk_xs = xs[:, :, cur:end]
        real_length = chunk_xs.shape[2]
        if real_length < chunk_size:
            chunk_xs = np.pad(chunk_xs, ((0, 0), (0, 0), (0, chunk_size - real_length)))
        x = {"mel": chunk_xs, "input_cache": input_cache}  #  ['mel', 'input_cache']

        out = model(x)  # ['wav', 'output_cache']
        wav = out["wav"]
        if cur == 0:
            shift_size = wav.shape[2] // chunk_xs.shape[2]
            xs = np.pad(
                xs, ((0, 0), (0, 0), (0, math.ceil(cumulative_delays / shift_size)))
            )
            logging.info(f"shift size {shift_size}")
        input_cache = out["output_cache"]
        outputs.append(wav)
        cur += chunk_size
    outputs = np.concatenate(outputs, axis=2)
    real_wav_length = real_mel_length * shift_size
    outputs = outputs[:, :, cumulative_delays : cumulative_delays + real_wav_length]
    return torch.from_numpy(outputs)


def main():
    """Run exporting process."""
    parser = argparse.ArgumentParser(description=("Export HiFiGAN model to ONNX. "))
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated wav files.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="onnx model file to be loaded.",
    )
    parser.add_argument(
        "--cumulative-delay",
        type=int,
        required=True,
        help="cumulative delay of the stream model. "
        "determines the starting point of the stream model.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=48, help="chunk size of the stream model."
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="sampling rate of generated wav",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ############################
    #       MEL2WAV CASE       #
    ############################
    # setup dataset
    assert args.dumpdir is not None
    format = check_format_in_directory(args.dumpdir)
    assert format is not None
    if format == "hdf5":
        mel_query = "*.h5"
        mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
    elif format == "npy":
        mel_query = "*-feats.npy"
        mel_load_fn = np.load

    else:
        raise ValueError("Support only hdf5 or npy format.")

    dataset = MelDataset(
        args.dumpdir,
        mel_query=mel_query,
        mel_load_fn=mel_load_fn,
        return_utt_id=True,
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = Model(
        model_path=args.onnx,
        providers=(
            ["CPUExecutionProvider"]
            if device == torch.device("cpu")
            else ["CUDAExecutionProvider"]
        ),
    )

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, items in enumerate(pbar, 1):
            utt_id, c = items
            f0, excitation = None, None
            batch = dict(normalize_before=False)
            if c is not None:
                c = torch.tensor(c, dtype=torch.float).to(device)
                batch.update(mel=c)
            start = time.time()
            y = stream_inference(model, batch, args.chunk_size, args.cumulative_delay).view(-1)
            rtf = (time.time() - start) / (len(y) / args.sampling_rate)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(args.outdir, f"{utt_id}_gen.wav"),
                y.cpu().numpy(),
                args.sampling_rate,
                "PCM_16",
            )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
