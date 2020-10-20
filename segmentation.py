from keras_segmentation.models.unet import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib

matplotlib.use("pdf")

M = resnet50_unet(n_classes=23)

M.train(train_images="original_images/", train_annotations="label_images_semantic/", checkpoints_path="resnet50_unet", epochs=30)

im = "original_images/004.jpg"
img_orig = Image.open(im)
validation_image = "label_images_semantic/004.png"
output = M.predict_segmentation(
    inp=im,
    out_fname="output.png"
)

fig, axs = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)
axs[0].imshow(img_orig)
axs[0].set_title('original image')
axs[0].grid(False)
axs[1].imshow(output)
axs[1].set_title('our model prediction')
axs[1].grid(False)
axs[2].imshow( Image.open(validation_image))
axs[2].set_title('initial masks')
axs[2].grid(False)
fig.savefig('wynik_modelu.png', dpi=300, bbox_inches='tight')