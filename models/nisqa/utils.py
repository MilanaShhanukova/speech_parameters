import os

import pandas as pd
import torch
import torch.nn as nn

from models.nisqa.lib import (NISQA, NISQA_DE, NISQA_DIM, SpeechQualityDataset,
                              predict_dim, predict_mos)

dev = None


def _getDevice(device: str = None):
    """
    Train on GPU if available.
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    if device is not None:
        if device == "cpu":
            dev = torch.device("cpu")
        elif device == "cuda":
            dev = torch.device("cuda")
    print("Device: {}".format(dev))
    return dev


def _loadModel(args, dev):
    """
    Loads the Pytorch models with given input arguments.
    """
    # if True overwrite input arguments from pretrained model
    if args["pretrained_model"]:
        checkpoint = torch.load(args["pretrained_model"], map_location=dev)

        # update checkpoint arguments with new arguments
        checkpoint["args"].update(args)
        args = checkpoint["args"]

    if args["model"] == "NISQA_DIM":
        args["dim"] = True
        args["csv_mos_train"] = None  # column names hardcoded for dim models
        args["csv_mos_val"] = None
    else:
        args["dim"] = False

    if args["model"] == "NISQA_DE":
        args["double_ended"] = True
    else:
        args["double_ended"] = False
        args["csv_ref"] = None

    # Load Model
    model_args = {
        "ms_seg_length": args["ms_seg_length"],
        "ms_n_mels": args["ms_n_mels"],
        "cnn_model": args["cnn_model"],
        "cnn_c_out_1": args["cnn_c_out_1"],
        "cnn_c_out_2": args["cnn_c_out_2"],
        "cnn_c_out_3": args["cnn_c_out_3"],
        "cnn_kernel_size": args["cnn_kernel_size"],
        "cnn_dropout": args["cnn_dropout"],
        "cnn_pool_1": args["cnn_pool_1"],
        "cnn_pool_2": args["cnn_pool_2"],
        "cnn_pool_3": args["cnn_pool_3"],
        "cnn_fc_out_h": args["cnn_fc_out_h"],
        "td": args["td"],
        "td_sa_d_model": args["td_sa_d_model"],
        "td_sa_nhead": args["td_sa_nhead"],
        "td_sa_pos_enc": args["td_sa_pos_enc"],
        "td_sa_num_layers": args["td_sa_num_layers"],
        "td_sa_h": args["td_sa_h"],
        "td_sa_dropout": args["td_sa_dropout"],
        "td_lstm_h": args["td_lstm_h"],
        "td_lstm_num_layers": args["td_lstm_num_layers"],
        "td_lstm_dropout": args["td_lstm_dropout"],
        "td_lstm_bidirectional": args["td_lstm_bidirectional"],
        "td_2": args["td_2"],
        "td_2_sa_d_model": args["td_2_sa_d_model"],
        "td_2_sa_nhead": args["td_2_sa_nhead"],
        "td_2_sa_pos_enc": args["td_2_sa_pos_enc"],
        "td_2_sa_num_layers": args["td_2_sa_num_layers"],
        "td_2_sa_h": args["td_2_sa_h"],
        "td_2_sa_dropout": args["td_2_sa_dropout"],
        "td_2_lstm_h": args["td_2_lstm_h"],
        "td_2_lstm_num_layers": args["td_2_lstm_num_layers"],
        "td_2_lstm_dropout": args["td_2_lstm_dropout"],
        "td_2_lstm_bidirectional": args["td_2_lstm_bidirectional"],
        "pool": args["pool"],
        "pool_att_h": args["pool_att_h"],
        "pool_att_dropout": args["pool_att_dropout"],
    }

    if args["double_ended"]:
        model_args.update(
            {
                "de_align": args["de_align"],
                "de_align_apply": args["de_align_apply"],
                "de_fuse_dim": args["de_fuse_dim"],
                "de_fuse": args["de_fuse"],
            }
        )

    print("Model architecture: " + args["model"])
    if args["model"] == "NISQA":
        model = NISQA(**model_args)
    elif args["model"] == "NISQA_DIM":
        model = NISQA_DIM(**model_args)
    elif args["model"] == "NISQA_DE":
        model = NISQA_DE(**model_args)
    else:
        raise NotImplementedError("Model not available")

    # Load weights if pretrained model is used ------------------------------------
    if args["pretrained_model"]:
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        print("Loaded pretrained model from " + args["pretrained_model"])
        if missing_keys:
            print("missing_keys:")
            print(missing_keys)
        if unexpected_keys:
            print("unexpected_keys:")
            print(unexpected_keys)
    return model, args


def _loadDatasetsFile(args):
    data_dir = os.path.dirname(args["deg"])
    file_name = os.path.basename(args["deg"])
    df_val = pd.DataFrame([file_name], columns=["deg"])

    # creating Datasets ---------------------------------------------------
    ds_val = SpeechQualityDataset(
        df_val,
        df_con=None,
        data_dir=data_dir,
        filename_column="deg",
        mos_column="predict_only",
        seg_length=args["ms_seg_length"],
        max_length=args["ms_max_segments"],
        to_memory=None,
        to_memory_workers=None,
        seg_hop_length=args["ms_seg_hop_length"],
        transform=None,
        ms_n_fft=args["ms_n_fft"],
        ms_hop_length=args["ms_hop_length"],
        ms_win_length=args["ms_win_length"],
        ms_n_mels=args["ms_n_mels"],
        ms_sr=args["ms_sr"],
        ms_fmax=args["ms_fmax"],
        ms_channel=args["ms_channel"],
        double_ended=args["double_ended"],
        dim=args["dim"],
        filename_column_ref=None,
    )
    return ds_val


def predict(args, model, ds_val, dev):
    if args["tr_parallel"]:
        model = nn.DataParallel(model)

    if args["dim"] == True:
        y_val_hat, y_val = predict_dim(
            model, ds_val, args["tr_bs_val"], dev, num_workers=args["tr_num_workers"]
        )
    else:
        y_val_hat, y_val = predict_mos(
            model, ds_val, args["tr_bs_val"], dev, num_workers=args["tr_num_workers"]
        )
    # print(y_val)
    if args["output_dir"]:
        ds_val.df["model"] = args["name"]
        ds_val.df.to_csv(
            os.path.join(args["output_dir"], "NISQA_results.csv"), index=False
        )
    return ds_val.df, y_val, y_val_hat
