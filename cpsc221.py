import argparse

parser = argparse.ArgumentParser(description="The diffusion script for our CPSC221 project.")

parser.add_argument("prompt_a", type=str, help="The prompt for the untransformed image")
parser.add_argument("prompt_b", type=str, help="The prompt for the transformed image")
parser.add_argument("-s", "--squares", type=int, help="The number of squares", default=4)
parser.add_argument("-i", "--iterations", type=int, help="The number of iterations of the distillation loop", default=1000)

args = parser.parse_args()

prompt_a = args.prompt_a
prompt_b = args.prompt_b
NUMBER_OF_SQUARES = args.squares
NUM_ITER = args.iterations

print("Importing packages...")

from rp import *
import rp
import torch
import source.stable_diffusion as sd
from source.learnable_textures import LearnableImageFourier
from source.stable_diffusion_labels import NegativeLabel
import time

#Optional: Specify what you DON'T want to see
negative_prompt = 'blurry ugly'

print()
print('Chosen prompts:')
print('    prompt_a =', repr(prompt_a)) #This will be right-side up
print('    prompt_b =', repr(prompt_b)) #This will be upside-down

if 's' not in dir():
    #You can select the original Stable Diffusion 1.5 or some dreambooth of it
    # model_name="CompVis/stable-diffusion-v1-4"
    model_name="runwayml/stable-diffusion-v1-5"
    # model_name="nitrosocke/Arcane-Diffusion"
    gpu=rp.select_torch_device()
    s=sd.StableDiffusion(gpu,model_name)
device=s.device

print("Chose device",s.device)

label_a = NegativeLabel(prompt_a,negative_prompt)
label_b = NegativeLabel(prompt_b,negative_prompt)

#Image Parametrization and Initialization (this section takes vram)

#Select Learnable Image Size (this has big VRAM implications!):
#Note: We use implicit neural representations for better image quality
#They're previously used in our paper "TRITON: Neural Neural Textures make Sim2Real Consistent" (see tritonpaper.github.io)
# ... and that representation is based on Fourier Feature Networks (see bmild.github.io/fourfeat)
learnable_image_maker = lambda: LearnableImageFourier(height=256,width=256,num_features=256,hidden_dim=256,scale=10).to(s.device);SIZE=256

image=learnable_image_maker()

import torch
import torch.nn.functional as F

def rotate_tiles(image, num_divisions=NUMBER_OF_SQUARES):
    image=as_torch_image(image)
    # Assuming image is a tensor of shape (num_channels, height, width)
    num_channels, height, width = image.shape

    tile_size=width//num_divisions

    # Calculate the number of tiles in each dimension
    tiles_x = width // tile_size
    tiles_y = height // tile_size

    # Initialize an output tensor
    output = torch.zeros_like(image)

    for x in range(tiles_x):
        for y in range(tiles_y):
            # Extract the tile
            tile = image[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]

            # Check if the tile should be rotated 90 or -90 degrees (checker pattern)
            if (x + y) % 2 == 0:
                # Rotate 90 degrees
                tile = tile.rot90(1, [1, 2])
            else:
                # Rotate -90 degrees
                tile = tile.rot90(-1, [1, 2])

            # Place the rotated tile back in the output tensor
            output[:, y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size] = tile

    return output

learnable_image_a=lambda: image()
learnable_image_b=lambda: rotate_tiles(image()) #Upside-down

optim=torch.optim.SGD(image.parameters(),lr=1e-4)

labels=[label_a,label_b]
learnable_images=[learnable_image_a,learnable_image_b]

#The weight coefficients for each prompt. For example, if we have [0,1], then only the upside-down mode will be optimized
weights=[1,1]

weights=rp.as_numpy_array(weights)
weights=weights/weights.sum()
weights=weights*len(weights)

#For saving a timelapse
ims=[]

def get_display_image():
    return rp.tiled_images(
        [
            rp.as_numpy_image(learnable_image_a()),
            rp.as_numpy_image(learnable_image_b()),
        ],
        length=len(learnable_images),
        border_thickness=0,
    )


#Set the minimum and maximum noise timesteps for the dream loss (aka score distillation loss)
s.max_step=MAX_STEP=990
s.min_step=MIN_STEP=10

display_eta=rp.eta(NUM_ITER, title='Status')

DISPLAY_INTERVAL = max(1, min(NUM_ITER//10, 50))

print('Every %i iterations we display an image in the form [image_a, image_b], where'%DISPLAY_INTERVAL)
print('    image_a = (the untransformed image)')
print('    image_b = (image_a, but transformed)')
print()
print('Interrupt the kernel at any time to return the currently displayed image')
print('You can run this cell again to resume training later on')
print()
print('Please expect this to take quite a while to get good images (>1h on my Mac). The longer you wait the better they\'ll be.')

try:
    for iter_num in range(NUM_ITER):
        display_eta(iter_num) #Print the remaining time

        preds=[]
        for label,learnable_image,weight in rp.random_batch(list(zip(labels,learnable_images,weights)), batch_size=1):
            pred=s.train_step(
                label.embedding,
                learnable_image()[None],

                #PRESETS (uncomment one):
                noise_coef=.1*weight,guidance_scale=100,#10
                # noise_coef=0,image_coef=-.01,guidance_scale=50,
                # noise_coef=0,image_coef=-.005,guidance_scale=50,
                # noise_coef=.1,image_coef=-.010,guidance_scale=50,
                # noise_coef=.1,image_coef=-.005,guidance_scale=50,
                # noise_coef=.1*weight, image_coef=-.005*weight, guidance_scale=50,
            )
            preds+=list(pred)

        with torch.no_grad():
            if not iter_num%max(1, DISPLAY_INTERVAL//4):
                im = get_display_image()
                ims.append(im)

                if not iter_num%DISPLAY_INTERVAL:
                    rp.save_image(im, "output_incomplete.png")

        optim.step()
        optim.zero_grad()
except KeyboardInterrupt:
    print()
    print('Interrupted early at iteration %i'%iter_num)
    im = get_display_image()
    ims.append(im)
    rp.save_image(im, "output_incomplete.png")

print('Untransformed Image:')
rp.save_image(rp.as_numpy_image(learnable_image_a()), "output_untransformed.png")

print('Transformed Image:')
rp.save_image(rp.as_numpy_image(learnable_image_b()), "output_transformed.png")

def save_run(name):
    folder="output_%s"%name.replace(" ", "_")
    if rp.path_exists(folder):
        folder+='_%i'%time.time()
    rp.make_directory(folder)
    ims_names=['ims_%04i.png'%i for i in range(len(ims))]
    with rp.SetCurrentDirectoryTemporarily(folder):
        rp.save_images(ims,ims_names,show_progress=True)
    print()
    print('Saved timelapse to folder:',repr(folder))

save_run('-'.join([prompt_a,prompt_b])) #You can give it a good custom name if you want!