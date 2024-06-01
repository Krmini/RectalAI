import torch

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from pathlib import Path
import imageio
from tqdm import tqdm
from IPython.display import display, Image


def to_gpu(ngpu):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return device

def loss_plot(G_losses, D_losses, out):
    x_values = range(1, len(G_losses) + 1)
    y_limit_G = max(G_losses[5:]) * 1.05
    y_limit_D = max(D_losses[5:]) * 1.05
    
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(x_values, G_losses, label="G")
    plt.plot(x_values, D_losses, label="D")
    plt.ylim(0, max(y_limit_G, y_limit_D))
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out + 'loss_plot.png')
    np.savetxt(fname = out+'loss_list.csv', X = np.column_stack((G_losses, D_losses)),
         fmt = '%.6f', delimiter=';', header='Generator_loss;Discriminator_loss')
    lowest_values = sorted(enumerate(G_losses), key=lambda x: x[1])[:10]
    for i, (iteration, value) in enumerate(lowest_values, 1):
        print(f"{i}. Iteration {iteration + 1}: {value}")


def image_grid(dataloader, img_list, device, out, wide_images=False):

    real_batch = next(iter(dataloader))
    if wide_images == False:
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
                   :64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig(out + 'real_fake_grid.png')
        plt.show()
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
        axes[0].imshow(np.transpose(vutils.make_grid(real_batch[0][:8].to(device), padding=5, normalize=True, nrow=1).cpu(), (1, 2, 0)))
        axes[0].set_title("Real Images")
        axes[0].axis("off")
        axes[1].imshow(np.transpose(vutils.make_grid(img_list[-1][:8].to(device), padding=5, normalize=True, nrow=1).cpu(), (1, 2, 0)))
        axes[1].set_title("Fake Images")
        axes[1].axis("off")
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(out + 'real_fake_grid.png')
        plt.show()

def create_gif (image_path):
    image_path = Path(image_path)
    gif_path = image_path/Path('Gan.gif')
    filenames = list(image_path.glob('*img.png'))
    filenames = sorted(filenames)

    # Function to add progress bar to the image
    def add_progress_bar(image, progress_percentage):
        # Calculate width of the progress bar
        progress_width = int(image.shape[1] * progress_percentage / 100)
        
        # Create a progress bar with the same height and width as the image
        progress_bar = np.zeros_like(image)
        
        # Set color of the progress bar to maximum intensity for the completed portion
        progress_bar[:10, :progress_width] = 255
        
        # Concatenate the progress bar with the image horizontally
        image_with_progress = np.concatenate([image, progress_bar], axis=0)
        
        return image_with_progress

    # Initialize image writer
    with imageio.get_writer(gif_path, mode='I') as writer:
        # Initialize tqdm to create a progress bar
        progress_bar = tqdm(total=len(filenames), desc='Creating GIF', unit='image')
        
        # Iterate through filenames
        for idx, filename in enumerate(filenames):
            image = imageio.imread(filename)
            
            # Calculate progress percentage
            progress_percentage = int((idx + 1) / len(filenames) * 100)
            
            # Add progress information to the image
            image_with_progress = add_progress_bar(image, progress_percentage)
            
            # Append image with progress bar to the GIF
            for i in range(10):
                writer.append_data(image_with_progress)
            
            # Update progress bar description
            progress_bar.set_postfix({'Progress': progress_percentage})
            progress_bar.update(1)
        print('Saving gif ....')

    # Finalize progress bar
    progress_bar.close()
    print('Gif saved')
    return gif_path

def show_gif(gif_path):
    display(Image(filename=gif_path))
