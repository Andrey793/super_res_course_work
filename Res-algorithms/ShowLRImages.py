import math
import matplotlib.pyplot as plt


def show_lr_images(images):
    """
    Displays a set of low-resolution images provided in a list.

    Args:
    - images: List of low-resolution images.
    """
    num_images = len(images)
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axs = axs.flatten()  # Flatten for easy indexing if axs is multidimensional

    for idx in range(num_rows * num_cols):
        if idx < num_images:
            axs[idx].imshow(images[idx], cmap='gray')
            axs[idx].set_title(f'Low-Res Image No.{idx + 1}')
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()