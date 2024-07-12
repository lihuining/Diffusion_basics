import torch
import torch.nn as nn
import open_clip
from torchvision import transforms
class Image_adapter(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mask = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feature):
        out_feature = self.adapter( self.sigmoid(self.mask)*feature ) + self.sigmoid(self.mask)*feature

        return out_feature   

def cal_cos(text, img, cos):
    a = text.mean(dim=1) # (bs,77,1024)->(bs,1024)
    b = img.squeeze(0) # (bs,1,1024)->(1,1024)
    sim = cos(a, b).mean()
    return sim
from PIL import Image 
if __name__ == "__main__":
    img_adapter = Image_adapter()
    dummy_input = Image.open("/mnt/workspace/workgroup_share/lhn/diffusion_basics/DisenBooth/dog7/00.jpg")
    size = 224
    clip_trans = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
                transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR ),
            ]
        )
    
    img_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='/mnt/workspace/workgroup_share/lhn/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin') 
    print(clip_trans(dummy_input).shape)
    with torch.no_grad():
        img_state = img_model.encode_image(clip_trans(dummy_input).unsqueeze(0)).unsqueeze(1) # [1,1,1024]
    print(img_state.shape)
    img_state = img_adapter(img_state) # [1,1,1024]
    print(img_state.shape)

