import torchvision
import torch,os,sys
import numpy as np
from PIL import Image

def read_startwith(path,name):
    turn = torchvision.transforms.ToTensor()
    result = []
    names = []
    print(path)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(name) and file.endswith("png"):
                path = os.path.join(root,file)
                names.append(file)
                image = Image.open(path).convert("L")
                image = turn(image)
                result.append(image)
    return torch.stack(result,0),names

def compute_mean_and_std(T):
    mean,std = T.mean([0,2,3],keepdim=True),T.std([0,2,3],keepdim=True)
    return mean,std

def write_image(path,images,names):
    if not os.path.exists(path):
        os.makedirs(path)
    turn = torchvision.transforms.ToPILImage()
    for i in range(len(names)):
        image = images[i]
        name = names[i]
        subpath = os.path.join(path,name)
        image = turn(image)
        image.save(subpath)

def turn_generate_image(g_path1,t_path2):
    origin_image,origin_name = read_startwith(t_path2,"image")
    generate_image,generate_name = read_startwith(g_path1,"image")
    origin_mean,origin_std = compute_mean_and_std(origin_image)
    generate_mean,generate_std = compute_mean_and_std(generate_image)
    print(origin_mean,origin_std,generate_mean,generate_std)
    generate_image = (generate_image - generate_mean) / generate_std * origin_std + origin_mean
    generate_image = torch.clip(generate_image,0,1)
    write_image("./fix_outputs/",generate_image,generate_name)

def turn_generate_mask(g_path1):
    generate_image,generate_name = read_startwith(g_path1,"mask")
    generate_image = (generate_image>0.5).float()
    write_image("fix_outputs",generate_image,generate_name)

if __name__ == "__main__":
    turn_generate_image("./outputs3/","./origin/")
    turn_generate_mask("./outputs3/")


