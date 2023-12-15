import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn import metrics


def evaluate(
    encoder: nn.Module,
    finetuned_head: nn.Module,
    test_dataset: Dataset,
    dataset_name: str,
    audio_length: int,
    device,
) -> dict:
    est_array = []
    gt_array = []
    clip_id_order = []

    encoder = encoder.to(device)
    encoder.eval()

    if finetuned_head is not None:
        finetuned_head = finetuned_head.to(device)
        finetuned_head.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            clip_id, others, label = test_dataset[idx]
            batch = test_dataset.concat_clip(idx, audio_length)
            batch = batch.to(device)

            output = encoder(batch)
            if finetuned_head:
                output = finetuned_head(output)

            # we always return logits, so we need a sigmoid here for multi-label classification
            if dataset_name in ["magnatagatune", "msd"]:
                output = torch.sigmoid(output)
            else:
                output = F.softmax(output, dim=1)

            track_prediction = output.mean(dim=0)
            clip_id_order.append(clip_id)
            est_array.append(track_prediction)
            gt_array.append(label)

    if dataset_name in ["magnatagatune", "msd"]:
        est_array = torch.stack(est_array, dim=0).cpu().numpy()
        gt_array = torch.stack(gt_array, dim=0).cpu().numpy()
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average="macro")
        #accuracy = metrics.accuracy_score(gt_array, est_array)
        #return {"GT_Array" : gt_array, "EST_Array" : est_array}

        # 기존 출력 결과에, accuaracy, gt label, prediction 추가 #

        return {
             "clip_id": clip_id_order,
             "PR-AUC": pr_aucs,
             "ROC-AUC": roc_aucs, 
             "GT_ARRAY": gt_array,
             "EST_ARRAY": est_array
         }

    est_array = torch.stack(est_array, dim=0)
    _, est_array = torch.max(est_array, 1)  # extract the predicted labels here.
    accuracy = metrics.accuracy_score(gt_array, est_array)
    return {"Accuracy": accuracy, "GT_Array" : gt_array, "EST_Array" : est_array}
