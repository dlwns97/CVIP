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
import random
from openai import OpenAI
import replicate

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

def training(client):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": f"make sentence like female singer singing calm ballad while playing piano if words 'female', 'ballad', 'piano', 'calm' is input you understand?"}
        ]
    )

def word_to_sentence(client, words):

    # w/ords = ['female', 'ballad/', 'piano', 'calmful']

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": f"make short one sentence with using exact these {words}"}
        ]
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    parser = Trainer.add_argparse_args(parser)

    config = yaml_config_hook("/home/seunghoi/clmr/config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    args.accelerator = None

    # args.checkpoint_path = './clmr_checkpoint_1550.pt'
    args.checkpoint_path = '/home/seunghoi/clmr/checkpoint_1000/best-model-epoch=1414-Train/loss=4.49.ckpt'
    #args.checkpoint_path = '/home/seunghoi/clmr/clmr_checkpoint_10000.pt'
    args.gpus = 4
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")
    print(args.checkpoint_path)
    train_transform = [RandomResizedCrop(n_samples=args.audio_length)]

    
    client = OpenAI(api_key="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")
    # REPLICATE_API_TOKEN = "r8_EjReI4xbiulHWm0P8BcLZn4F7AUEaxw3Zy2uM"
    training(client)

    # ------------
    # dataloaders
    # ------------
    # train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="train", download=False)
    # valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="valid", download=False)
    
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test", download=False)
    #test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test", download=False)
    
    # contrastive_train_dataset = ContrastiveDataset(
    #     train_dataset,
    #     input_shape=(1, args.audio_length),
    #     transform=Compose(train_transform),
    # )

    # contrastive_valid_dataset = ContrastiveDataset(
    #     valid_dataset,
    #     input_shape=(1, args.audio_length),
    #     transform=Compose(train_transform),
    # )

    contrastive_test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
        transform=None,
    )

    # train_loader = DataLoader(
    #     contrastive_train_dataset,
    #     batch_size=args.finetuner_batch_size,
    #     num_workers=args.workers,
    #     shuffle=True,
    # )

    # valid_loader = DataLoader(
    #     contrastive_valid_dataset,
    #     batch_size=args.finetuner_batch_size,
    #     num_workers=args.workers,
    #     shuffle=False,
    # )

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

    state_dict = load_encoder_checkpoint(args.checkpoint_path, test_dataset.n_classes)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    module = LinearEvaluation(
        args,
        cl.encoder,
        hidden_dim=n_features,
        output_dim=test_dataset.n_classes,
    )

    # train_representations_dataset = module.extract_representations(train_loader)
    # train_loader = DataLoader(
    #     train_representations_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     shuffle=True,
    # )

    # valid_representations_dataset = module.extract_representations(valid_loader)
    # valid_loader = DataLoader(
    #     valid_representations_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     shuffle=False,
    # )
    args.finetuner_checkpoint_path = '/home/seunghoi/clmr/linear_module.ckpt'
    #args.finetuner_checkpoint_path = '/home/seunghoi/clmr/linear_module_original.ckpt'
    if args.finetuner_checkpoint_path:
        state_dict = load_finetuner_checkpoint(args.finetuner_checkpoint_path)
        module.model.load_state_dict(state_dict)
    # else:
    #     early_stop_callback = EarlyStopping(
    #         monitor="Valid/loss", patience=10, verbose=False, mode="min"
    #     )

    #     trainer = Trainer.from_argparse_args(
    #         args,
    #         logger=TensorBoardLogger(
    #             "runs", name="CLMRv2-eval-{}".format(args.dataset)
    #         ),
    #         max_epochs=args.finetuner_max_epochs,
    #         callbacks=[early_stop_callback],
    #     )
    #     trainer.fit(module, train_loader, valid_loader)
    #     torch.save(module.model.state_dict(), '/home/seunghoi/clmr/linear_module_original.ckpt')


    device = "cuda:0" if args.gpus else "cpu"
    results = evaluate(
        module.encoder,
        module.model,
        contrastive_test_dataset,
        args.dataset,
        args.audio_length,
        device=device,
    )
    

    # Evaluate 결과 출력 부분 #
    # AUC 랑 Accuracy #
    print("PR-AUC : ", results["PR-AUC"])
    print("ROC_AUC : ", results["ROC-AUC"])
    
    # TOP 5 LABEL EXTRACTION #

    # TAGS (CLASSES) #
#     tag = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic',
#  'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female',
#  'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord',
#  'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft',
#  'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance',
#  'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird',
#  'country', 'metal', 'female voice', 'choral']
#     tag = ['no vocal', 'guitar', 'classical', 'vocal', 'slow', 'techno', 'drum', 'electronic', 
#            'rock', 'fast', 'male', 'female', 'piano', 'ambient', 'violin', 'beat', 
#            'synth', 'indian', 'opera', 'harpsichord', 'loud', 'string', 'quiet', 'flute', 
#            'pop', 'soft', 'sitar', 'classic', 'choir', 'new age', 'dance', 'beats', 'harp', 
#            'cello', 'weird', 'country', 'metal', 'choral', 'electro', 'jazz', 'eastern', 
#            'instrumental', 'bass', 'modern', 'no piano', 'harpsicord', 'jazzy', 
#            'baroque', 'foreign', 'orchestra']

    # tag = ['no vocal', 'guitar', 'classic', 'vocal', 'slow', 'techno', 'drum', 'electro', 'rock', 
    # 'fast', 'male', 'beat', 'female', 'piano', 'ambient', 'violin', 'synth', 'indian', 'opera', 
    # 'harpsichord', 'loud', 'string', 'quiet', 'flute', 'pop', 'soft', 'sitar', 'choir', 'new age', 
    # 'dance', 'harp', 'cello', 'weird', 'country', 'metal', 'choral', 'jazz', 'eastern', 'bass', 
    # 'modern', 'jazzy', 'baroque', 'foreign', 'orchestra', 'hard rock', 'electric', 'trance', 'folk', 
    # 'chorus', 'chant']

    # 이게 마지막으로 사용한 finetuning할 때 사용한거
    tag = ['no vocal', 'vocal', 'slow', 'techno', 'electric', 'drum', 'rock', 'fast', 'male', 'female', 'piano', 'ambient', 'violin', 'synth', 'indian', 'opera', 'harpsichord', 'loud', 'string', 'silence', 'flute', 'pop', 'soft', 'sitar', 'choir', 'new age', 'dance', 'harp', 'chorus', 'cello', 'weird', 'jazzy', 'country', 'metal', 'eastern', 'bass', 'modern', 'chant', 'baroque', 'foreign', 'classic guitar','electric guitar','orchestra', 'hard rock', 'trance', 'funky', 'folk', 'spanish', 'heavy', 'upbeat']
    
    # 기존 top 50
    #tag = ['guitar' , 'classical' , 'slow' , 'techno' , 'strings' , 'drums' , 'electronic', 'rock' , 'fast' , 'piano' , 'ambient' , 'beat' , 'violin' , 'vocal' , 'synth' , 'female', 'indian' , 'opera' , 'male' , 'singing' , 'vocals' , 'no vocals' , 'harpsichord', 'loud' , 'quiet' , 'flute' , 'woman' , 'male vocal' , 'no vocal' , 'pop' , 'soft', 'sitar' , 'solo' , 'man' , 'classic' , 'choir' , 'voice' , 'new age' , 'dance', 'male voice' , 'female vocal' , 'beats' , 'harp' , 'cello' , 'no voice' , 'weird', 'country' , 'metal' , 'female voice' , 'choral']
    
    
    # print(len(tag))
    # classic + classical
    # male + male opera
    # beat + beats
    # electro + electronic
    # harpsichord + harpsicord

    
    GT_LABEL = results["GT_ARRAY"]
    PR_LABEL = results["EST_ARRAY"]
    CLIP_ID = results['clip_id']
    
    # TOP 5 인덱스를 뽑기
    
    TOP_5_GT = [np.argsort(labels)[-10:][::-1] for labels in GT_LABEL]
    TOP_5_PR = [np.argsort(labels)[-10:][::-1] for labels in PR_LABEL]
    TOP_5_SCORE = [PR_LABEL[score] for score in TOP_5_PR]
    # # GT LABEL #
    TOP_5_GT_CLASSES = []
    for indices in TOP_5_GT:
        TOP_5_CLASS = [tag[index] for index in indices]
        TOP_5_GT_CLASSES.append(TOP_5_CLASS)
    
    
    # GT_LABEL 배열의 각 요소에 대해 1인 값의 인덱스를 찾기
    GT_1_LABELS = [np.where(labels == 1)[0] for labels in GT_LABEL]
    #print(GT_1_LABELS)
    # 1인 값에 해당하는 클래스 이름 추출 및 출력
    GT_1_CLASSES = []
    for indices in GT_1_LABELS:
        GT_1_CLASS = [tag[index] for index in indices]
        GT_1_CLASSES.append(GT_1_CLASS)
        
        
    # PREDICTED LABEL #
    TOP_5_PR_CLASSES = []
    TOP_5_PR_SCORE = []
    for idx, indices in enumerate(TOP_5_PR):
        TOP_5_CLASS = [tag[index] for index in indices]
        TOP_5_PR_CLASSES.append(TOP_5_CLASS)
        TOP_5_PR_SCORE.append([PR_LABEL[idx][index] for index in indices])

    # TOP 5 라벨 출력 #
    # for i in [random.randint(0, 5329) for _ in range(10)]:

    idx_set = [random.randint(0, 5329) for _ in range(20)]
    #idx_set = [CLIP_ID.index(idx) for idx in ['57425','10381','2339','24635','21513','19963','12630','9043','48750','7550','52552','37827','2814','2683','8310','19589','20840','45092','53764','2368']]
    
    for i in idx_set:
        threshold_top = [score for score in TOP_5_PR_SCORE[i] if score >= 0.2]
        length = len(threshold_top)
        print("CLIP ID  : ", CLIP_ID[i])
        print("GT LABEL (1s) : ", GT_1_CLASSES[i])
        
        #print("GT LABEL : ", TOP_5_GT_CLASSES[i][:length])
        
        print("PR LABEL : ", TOP_5_PR_CLASSES[i][:length])
        print("PR SCORE : ", TOP_5_PR_SCORE[i][:length])
        sentence = word_to_sentence(client, TOP_5_PR_CLASSES[i][:length])
        print("sentence : ", sentence)
        # output = replicate.run(
        #     "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        #     input={"prompt": sentence}
        # )
        # print("link : ", output)
        print()