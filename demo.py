import numpy as np
import torch
import os

# os.system('! wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip')
# os.system('! unzip -d weights -j weights.zip')
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import imageio
import pickle

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)


# load and normalize image
# input_image = Image.open('example_image.jpg')
input_image = Image.open('/home/cc/students/csguests/chendudai/Thesis/data/0_1_undistorted/images/0053.jpg')
folder = '0053/'
names = ['_window', '_statue', '_door', '_facade', '_tower', '_top_part', '_spire']

# or load from URL...
# image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
# input_image = Image.open(requests.get(image_url, stream=True).raw)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])
img = transform(input_image).unsqueeze(0)


for name in names:
    if name == '_window':
        prompts = ['a window', 'windows', 'a photo of a window', 'a photo of windows', 'a photo of a window of the cathedral',
                   'a photo of windows of the cathedral', 'a window of a cathedral', 'The windows of the cathedral']
    if name == '_statue':
        prompts = ['statue', 'a photo of a statue', 'a statue of the cathedral', 'the statues of the cathedral',
                   'a photo of a statue of the cathedral', 'a photo of the statues of the cathedral']
    if name == '_door':
        prompts = ['door', 'a photo of a door', 'a door of the cathedral', 'the doors of the cathedral',
                   'a photo of a door of the cathedral', 'a photo of the doors of the cathedral']
    if name == '_facade':
        prompts = ['facade', 'a photo of a facade', 'a facade of the cathedral', 'the facade of the cathedral',
                   'a photo of a facade of the cathedral', 'a photo of the facade of the cathedral']
    if name == '_tower':
        prompts = ['tower', 'a photo of a tower', 'a tower of the cathedral', 'the tower of the cathedral',
                   'a photo of a tower of the cathedral', 'a photo of the tower of the cathedral']
    if name == '_top_part':
        prompts = ['top part', 'a photo of a top part', 'a top part of the cathedral', 'the top part of the cathedral',
                   'a photo of a top part of the cathedral', 'a photo of the top part of the cathedral']
    if name == '_spire':
        prompts = ['spire', 'a photo of a spire', 'a spire of the cathedral', 'the spires of the cathedral',
                   'a photo of a spire of the cathedral', 'a photo of the spires of the cathedral']

    threshold = 0.5
    threshold_str = str(threshold)

    # predict
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

    # show prediction
    _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    [ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))]
    [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))]
    plt.tight_layout()
    os.makedirs('./results/' + folder, exist_ok=True)
    plt.savefig('./results/' + folder + 'prediction' + name + '.png')
    plt.show()


    if name == '_window':
        for i, prompt in enumerate(prompts):
            with open('./results/' + folder + prompt.replace(" ", "_") + '.pkl', 'wb') as handle:
                mask = torch.sigmoid(preds[i][0])
                mask[mask<0.5] = 0
                mask[mask>=0.5] = 1
                pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # show prediction with threshold
    # _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    # [a.axis('off') for a in ax.flatten()]
    # ax[0].imshow(input_image)
    # for i in range(len(prompts)):
    #     x = torch.sigmoid(preds[i][0])
    #     x[x<threshold] = 0
    #     x[x>=threshold] = 1
    #     ax[i+1].imshow(x)
    #     ax[i+1].text(0, -15, prompts[i])
    # plt.tight_layout()
    # plt.savefig('./results/' + folder + 'prediction_threshold' + name + '_thershold_' + threshold_str + '.png')
    # plt.show()

    # show prediction with threshold overlay
    _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    input_image = input_image.resize((352, 352))
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img_tensor = transform(input_image) / 255
    for i in range(len(prompts)):
        x = torch.sigmoid(preds[i][0])
        x[x<threshold] = 0
        x[x>=threshold] = 1
        y = 0.2*img_tensor[0,:,:] + 0.8*x
        img_tensor[0, :, :] = img_tensor[0, :, :] * (1 - x)
        img_tensor[1, :, :] = img_tensor[1, :, :] * (1 - x)
        img_tensor[2, :, :] = img_tensor[2, :, :] * (1 - x) + (255 * x)
        ax[i+1].imshow(img_tensor.permute(1,2,0))
        ax[i+1].text(0, -15, prompts[i])
    plt.tight_layout()
    plt.savefig('./results/' + folder + 'prediction_overlay' + name + '_thershold_' + threshold_str + '.png')
    plt.show()






    threshold = 0.25
    threshold_str = str(threshold)
    # # show prediction with threshold
    # _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    # [a.axis('off') for a in ax.flatten()]
    # ax[0].imshow(input_image)
    # for i in range(len(prompts)):
    #     x = torch.sigmoid(preds[i][0])
    #     x[x<threshold] = 0
    #     x[x>=threshold] = 1
    #     ax[i+1].imshow(x)
    #     ax[i+1].text(0, -15, prompts[i])
    # plt.tight_layout()
    # plt.savefig('./results/' + folder + 'prediction_threshold' + name + '_thershold_' + threshold_str + '.png')
    # plt.show()

    # show prediction with threshold overlay
    _, ax = plt.subplots(1, len(prompts)+2, figsize=(25, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    input_image = input_image.resize((352, 352))
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img_tensor = transform(input_image) / 255
    for i in range(len(prompts)):
        x = torch.sigmoid(preds[i][0])
        x[x<threshold] = 0
        x[x>=threshold] = 1
        y = 0.2*img_tensor[0,:,:] + 0.8*x
        img_tensor[0, :, :] = img_tensor[0, :, :] * (1 - x)
        img_tensor[1, :, :] = img_tensor[1, :, :] * (1 - x)
        img_tensor[2, :, :] = img_tensor[2, :, :] * (1 - x) + (255 * x)
        ax[i+1].imshow(img_tensor.permute(1,2,0))
        ax[i+1].text(0, -15, prompts[i])
    plt.tight_layout()
    plt.savefig('./results/' + folder + 'prediction_overlay' + name + '_thershold_' + threshold_str + '.png')
    plt.show()
