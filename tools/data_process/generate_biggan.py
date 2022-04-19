import logging
import random
import numpy as np
import os
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample, one_hot_from_int,
                                       save_as_images, display_in_terminal, convert_to_images)
import torch

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
model = BigGAN.from_pretrained('biggan-deep-256')


# Prepare a input
truncation = 0.4
# class_vector = one_hot_from_names(
#     ['soap bubble', 'coffee', 'mushroom'], batch_size=3)
BATCH = 64
for k in range(16):
    class_vector = one_hot_from_int(
        random.sample(range(0, 1000), BATCH), batch_size=BATCH)
    noise_vector = truncated_noise_sample(
        truncation=truncation, batch_size=BATCH)

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to('cuda')
    class_vector = class_vector.to('cuda')
    model.to('cuda')

    # Generate an image
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    # If you have a sixtel compatible terminal you can display the images in the terminal
    # (see https://github.com/saitoha/libsixel for details)
    # display_in_terminal(output)

    # Save results as png images
    # save_as_images(output)

    img = convert_to_images(output)
    path = os.path.abspath("../../data/biggan_imagenet/")
    if not os.path.exists(path):
        os.makedirs(path)
    for i, out in enumerate(img):
        current_file_name = path + '/' + \
            '%d.png' % (k*BATCH + i)
        # logger.info("Saving image to {}".format(current_file_name))
        out.save(current_file_name, 'png')
