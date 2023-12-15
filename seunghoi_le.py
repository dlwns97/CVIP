import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchaudio_augmentations import Compose, RandomResizedCrop
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import torch

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.evaluation import evaluate
from clmr.models import SampleCNN, SampleCNN_1550
from clmr.modules import ContrastiveLearning, LinearEvaluation
from clmr.utils import (
    yaml_config_hook,
    load_encoder_checkpoint,
    load_finetuner_checkpoint,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    args.accelerator = None

    #args.checkpoint_path = './clmr_checkpoint_1550.pt'
    args.checkpoint_path = 'checkpoint_1000/best-model-epoch=1414-Train/loss=4.49.ckpt'
    args.gpus = 4
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")
    
    print(args.checkpoint_path)
    finetuner_path = '/home/seunghoi/clmr/linear_module.ckpt'
    if os.path.isfile(finetuner_path) == False:
        train_transform = [RandomResizedCrop(n_samples=args.audio_length)]
        
        train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train", download=False)
        valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid", download=False)

        contrastive_train_dataset = ContrastiveDataset(
            train_dataset,
            input_shape=(1, args.audio_length),
            transform=Compose(train_transform),
        )

        contrastive_valid_dataset = ContrastiveDataset(
            valid_dataset,
            input_shape=(1, args.audio_length),
            transform=Compose(train_transform),
        )


        train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=args.finetuner_batch_size,
            num_workers=args.workers,
            shuffle=True,
        )

        valid_loader = DataLoader(
            contrastive_valid_dataset,
            batch_size=args.finetuner_batch_size,
            num_workers=args.workers,
            shuffle=False,
        )

    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test", download=False)


    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
        transform=None,
    )

    test_loader = DataLoader(
        contrastive_test_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False,
    )

    # ------------
    # encoder
    # ------------
    encoder = SampleCNN(
         strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
         supervised=args.supervised,
         out_dim=test_dataset.n_classes,
    )
    #encoder = SampleCNN_1550(
    #    strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
    #    supervised=args.supervised,
    #    out_dim=train_dataset.n_classes,
    #)

    n_features = encoder.fc.in_features  # get dimensions of last fully-connected layer

    encoder_state_dict = load_encoder_checkpoint(args.checkpoint_path, test_dataset.n_classes)
    encoder.load_state_dict(encoder_state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    module = LinearEvaluation(
        args,
        cl.encoder,
        hidden_dim=n_features,
        output_dim=test_dataset.n_classes,
    )

    if os.path.isfile(finetuner_path):
        state_dict = load_finetuner_checkpoint(finetuner_path)
        module.model.load_state_dict(state_dict)
    else:
        early_stop_callback = EarlyStopping(
            monitor="Valid/loss", patience=10, verbose=False, mode="min"
        )

        trainer = Trainer.from_argparse_args(
            args,
            logger=TensorBoardLogger(
                "runs", name="CLMRv2-eval-{}".format(args.dataset)
            ),
            max_epochs=args.finetuner_max_epochs,
            callbacks=[early_stop_callback],
        )
        trainer.fit(module, train_loader, valid_loader)
        torch.save(module.model.state_dict(), '/home/seunghoi/clmr/linear_module.ckpt')

    device = "cuda:0" if args.gpus else "cpu"
    results = evaluate(
        module.encoder,
        module.model,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device=device,
    )
    
    # print(results)

    # Evaluate 결과 출력 부분 #
    # AUC 랑 Accuracy #
    print("clip_id : ", results['clip_id'])
    print("PR-AUC : ", results["PR-AUC"])
    print("ROC_AUC : ", results["ROC-AUC"])
    
    # TOP 5 LABEL EXTRACTION #

    # TAGS (CLASSES)
    tag = ['no vocal', 'classic', 'vocal', 'slow', 'techno', 'electric', 'drum', 'rock', 'fast', 'male', 'female', 'piano', 'ambient', 'violin', 'synth', 'indian', 'opera', 'harpsichord', 'loud', 'string', 'silence', 'flute', 'pop', 'soft', 'sitar', 'choir', 'new age', 'dance', 'harp', 'chorus', 'cello', 'weird', 'jazzy', 'country', 'metal', 'eastern', 'bass', 'modern', 'chant', 'baroque', 'classical guitar', 'foreign', 'orchestra', 'hard rock', 'trance', 'funky', 'folk', 'spanish', 'heavy', 'upbeat']

    # classic + classical
    # male + male opera
    # beat + beats
    # electro + electronic
    # harpsichord + harpsicord

    
    GT_LABEL = results["GT_ARRAY"]
    PR_LABEL = results["EST_ARRAY"]
    
    # TOP 5 인덱스를 뽑기
    TOP_5_GT = [np.argsort(labels)[-5:][::-1] for labels in GT_LABEL]
    TOP_5_PR = [np.argsort(labels)[-5:][::-1] for labels in PR_LABEL]

    # GT LABEL #
    TOP_5_GT_CLASSES = []
    for indices in TOP_5_GT:
        TOP_5_CLASS = [tag[index] for index in indices]
        TOP_5_GT_CLASSES.append(TOP_5_CLASS)

    # PREDICTED LABEL #
    TOP_5_PR_CLASSES = []
    for indices in TOP_5_PR:
        TOP_5_CLASS = [tag[index] for index in indices]
        TOP_5_PR_CLASSES.append(TOP_5_CLASS)

    # TOP 5 라벨 출력 #
    for i in range(3):
        print("GT LABEL : ", TOP_5_GT_CLASSES[i])
        print("PR LABEL : ", TOP_5_PR_CLASSES[i])
        print()
    



