from .get_mir_dataloader import get_mir_loaders
from .get_fma_dataloader import get_fma_loaders
from .get_mtt_dataloader import get_mtt_loaders

from data.vision import get_deepscores_dataloader 
from data.vision import get_universal_dataloader

def get_dataset(args):
    if args.domain == "audio":
        if args.dataset == "billboard":
            (train_loader, train_dataset, test_loader, test_dataset) = get_mir_loaders(args)
        elif args.dataset == "fma":
            (train_loader, train_dataset, test_loader, test_dataset) = get_fma_loaders(args)
        elif args.dataset == "magnatagatune":
            (train_loader, train_dataset, test_loader, test_dataset) = get_mtt_loaders(args)
        else:
            raise NotImplementedError
    elif args.domain == "vision":
        if args.dataset == "deepscores":
            (train_loader, train_dataset, test_loader, test_dataset) = get_deepscores_dataloader(args)
        elif args.dataset == "universal":
            (train_loader, train_dataset, test_loader, test_dataset) = get_universal_dataloader(args)
        else:
            raise NotImplementedError
    return train_loader, train_dataset, test_loader, test_dataset