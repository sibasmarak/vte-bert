import clip, time
import pickle, os
import timm, torch
from PIL import Image
import numpy as np, time, scipy
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_img_embeddings(data_dir, split='dev', modelname = 'vgg16_bn', data='snli'):
    start = time.time()

    split_dir = os.path.join(data_dir, f'{split}.pkl')

    model, preprocess = clip.load(modelname, device=device)
    model.float()

    # model = timm.create_model(modelname, pretrained=True)
    # model.reset_classifier(0)
    model = model.to(device)

    with open(split_dir, 'rb') as f:
        imgs = pickle.load(f)['image_names']
    #     for a in imgs:
    #         if len(a) < 2:
    #             print(a, len(a))
    #             time.sleep(3)
        

    print(f"Number of images: {len(imgs)}")
    features = []
    for i, img in enumerate(imgs):
        if (i+1) % 100 == 0:
            end = time.time()
            print(f'Completed: [{i + 1}|{len(imgs)}] Time taken: {(end - start) // 60} m {(end - start) % 60:.4f} s')
        
        if data == 'snli': 
            orig_im = Image.open(f'../flickr30k-images/{img}')
        elif data == 'tweet':
            orig_im = Image.open(f'../{img}')

        im = preprocess(orig_im).unsqueeze(0).to(device)
        # the following lines are not required for clip-RN50
        # try:
        #     im = orig_im.resize((224, 224))
        #     im = np.array(im).reshape((3, 224, 224))
        #     im = im.astype('float32')
        #     im /= 255

        #     im = torch.tensor(im).unsqueeze(0)
        #     im = im.to(device)
        # except:
        #     features.append(feature)
        #     continue


        with torch.no_grad():
            feature = model.encode_image(im)
        
        
        # feature = model(im)

        feature = feature.cpu().detach().numpy()

        features.append(feature)

    model_dir = os.path.join(data_dir, f'{modelname}')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, f'{split}.npy'), 'wb') as f:
        np.save(f, features)


data_dir = '../data/vte/'
split = 'dev'
modelname = 'RN50'
data = 'snli'
create_img_embeddings(data_dir, split=split, modelname=modelname, data=data)