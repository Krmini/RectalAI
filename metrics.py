import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from piq.feature_extractors import InceptionV3
from piq import FID

def FID_plot(real_dataloader, generated_img_list, save_interval, out):

    def normalise_batch(batch):
        mean = batch.mean(dim=(0, 2, 3))
        std = batch.std(dim=(0, 2, 3))
        normalize = transforms.Normalize(mean=mean, std=std)
        normalized_batch = normalize(batch)
        scale_factor = 1 / torch.max(torch.tensor(1e-5), std)
        normalized_batch = torch.clamp(normalized_batch * scale_factor.view(1, -1, 1, 1), 0, 1)
        return normalized_batch

    def compute_feats(
        loader: torch.utils.data.DataLoader,
        feats,
        device: str = 'cuda') -> torch.Tensor:
 
        feature_extractor = InceptionV3()
        feature_extractor.to(device)
        feature_extractor.eval()

        if feats == 'real':
            total_feats = []
            for batch in tqdm(real_dataloader, desc = 'Extracting real features'):
                images = batch[0]
                images = normalise_batch(images)

                #Ensure "RGB" images for InceptionV3 net
                if images.shape[1] == 1:
                    images = torch.cat([images, images, images], dim=1)

                N = images.shape[0]
                images = images.float().to(device)
    
                # Get features
                features = feature_extractor(images)
                assert len(features) == 1, \
                    f"feature_encoder must return list with features from one layer. Got {len(features)}"
    
                features = features[0].view(N, -1)
                features = features.cpu()
                total_feats.append(features)
                torch.cuda.empty_cache()
        else:
            total_feats = []
            images = loader
            
            images = normalise_batch(images)

            #Ensure "RGB" images for InceptionV3 net
            if images.shape[1] == 1:
                images = torch.cat([images, images, images], dim=1)

            N = images.shape[0]
            images = images.float().to(device)
    
            # Get features
            features = feature_extractor(images)
            # TODO(jamil 26.03.20): Add support for more than one feature map
            assert len(features) == 1, \
                f"feature_encoder must return list with features from one layer. Got {len(features)}"
    
            features = features[0].view(N, -1)
            features = features.cpu()
            total_feats.append(features)
            torch.cuda.empty_cache()
            

        feature_extractor.cpu()
        torch.cuda.empty_cache()
        return torch.cat(total_feats, dim=0)

    fid_metric = FID()
    first_feats = compute_feats(loader = real_dataloader, feats='real')
    
    FID = []
    iterations = []
    for i, images in enumerate(tqdm(generated_img_list, desc = 'Extracting generated features')):
        iterations.append(save_interval*(i+1))
        second_feats = compute_feats(loader = images, feats = 'fake')
        fid = fid_metric(first_feats, second_feats)
        FID.append(float(fid))
    
    plt.figure(figsize=(10, 5))
    plt.title("Real and fake images Fréchet inception distance (FID) during Training")
    plt.plot(iterations, FID, label="FID")
    plt.xlabel("iterations")
    plt.ylabel("FID")
    plt.legend()
    plt.savefig(out + 'FID_plot.png')
    plt.show()