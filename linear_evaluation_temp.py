import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from clmr.datasets import get_dataset
from clmr.data import ContrastiveDataset
from clmr.models import SampleCNN
from clmr.modules import ContrastiveLearning
from clmr.utils import yaml_config_hook, load_encoder_checkpoint



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

    args.checkpoint_path = '/home/seunghoi/clmr/checkpoint_1000/best-model-epoch=1414-Train/loss=4.49.ckpt'
    args.gpus = 4
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError("That checkpoint does not exist")

    # 라벨이 없는 데이터셋 로드
    test_dataset = get_dataset(args.dataset, args.dataset_dir, subset="test", download=False)
    test_dataset = ContrastiveDataset(
        test_dataset,
        input_shape=(1, args.audio_length),
        transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.finetuner_batch_size,
        num_workers=args.workers,
        shuffle=False
    )

    # 모델 설정
    encoder = SampleCNN(
         strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
         supervised=args.supervised,
         out_dim=test_dataset.n_classes,
    )

    n_features = encoder.fc.in_features
    state_dict = load_encoder_checkpoint(args.checkpoint_path, test_dataset.n_classes)
    encoder.load_state_dict(state_dict)

    cl = ContrastiveLearning(args, encoder)
    cl.eval()
    cl.freeze()

    device = "cuda:0" if args.gpus else "cpu"

    # 라벨이 없는 데이터에 대한 예측 수행
    for batch in test_loader:
        inputs, _ = batch
        inputs = inputs.to(device)
        outputs = cl.encoder(inputs)
        predicted_labels = torch.sigmoid(outputs).detach().cpu().numpy()
        
        # 예측 라벨 처리
        # 예를 들어 출력하거나 저장하는 코드를 여기에 추가
