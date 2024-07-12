from typing import Union, Tuple, List, Callable, Dict, Optional
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
import numpy as np
from IPython.display import display
from PIL import Image
import abc 
import ptp_utils
import seq_aligner


class LocalBlend: # local edit,添加mask之后进行局部修改
    '''
    local_blend(x_t, self.attention_store, self.cur_step)
    '''
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold: float = .3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold

    def __call__(self, x_t, attention_store, step):
        k = 1
        maps = attention_store["down_cross"][:2] + attention_store["up_cross"][3:6]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = F.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       



class AttentionControl(abc.ABC): # ABC:抽象基类
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1 ### net_.__class__.__name__ == 'CrossAttention', cross attn 层数,在register_attention_control当中初始化
        self.cur_att_layer = 0
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @abc.abstractmethod # 抽象方法定义
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str): ###
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

class EmptyControl(AttentionControl): # 在AttentionControl基础上没有作用，知识实现抽象函数进行继承
    # 对attn不进行修改
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):
    # 保存每个step的attention map总和,
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store() # 初始化注意力层的dict,保存单步attention map
        self.attention_store = {}
    @staticmethod # 静态方法,没有self参数,可以直接通过类进行调用
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str): # 根据attention类型和位置加入到self.step_store当中
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0: # 初始化
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store() # 对self.step_store置空

    def get_average_attention(self): # len(self.attention_store['down_cross']) = 4 ??
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}



        
class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1]) # （0,10）
        self.local_blend = local_blend
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2: ### ???
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet) # 调用父类AttentionStore的forward函数
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]): # soft attention constrain
            h = attn.shape[0] // (self.batch_size) # 32/4 = 8
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:]) # 4, 8, 1024, 1024
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    


class AttentionReplace(AttentionControlEdit):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device) # (3,77,77)

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      

        

class AttentionRefine(AttentionControlEdit):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace



def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    # 用于reweighting
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        for i in inds:
            equalizer[:, i] = values
    return equalizer

class AttentionReweight(AttentionControlEdit):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2  # 16^^2
    for location in from_where:
        print(attention_maps.keys())
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels: # [8,256,77]
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps) # [8,16,16,77]
    out = torch.cat(out, dim=0) # len = 5,list每个元素大小为[8,16,16,77]--[40, 16, 16, 77]
    out = out.sum(0) / out.shape[0]
    return out.cpu() # [16,16,77]

# show_cross_attention(controller, res=16, from_where=["down"])
def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0): # select表示选择哪个prompt进行可视化
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)): # 遍历每个有效token
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3) # 扩充为三通道
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def sort_by_eq(eq):
    
    def inner_(images):
        swap = 0
        if eq[-1] < 1:
            for i in range(len(eq)):
                if eq[i] > 1 and eq[i + 1] < 1:
                    swap = i + 2
                    break
        else:
             for i in range(len(eq)):
                if eq[i] < 1 and eq[i + 1] > 1:
                    swap = i + 2
                    break
        print(swap)
        if swap > 0:
            images = np.concatenate([images[1:swap], images[:1], images[swap:]], axis=0)
            
        return images
    return inner_


def run_and_display(prompts, controller, latent=None, run_baseline=True, callback:Optional[Callable[[np.ndarray], np.ndarray]] = None, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent,_ = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False)
        print("results with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm(ldm, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator)
    if callback is not None:
        images = callback(images)
    results = ptp_utils.save_images(images)
    return images, x_t,results

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model_id = "/mnt/workspace/workgroup_share/lhn/utils/CompVis/ldm-text2im-large-256"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 5.
MAX_NUM_WORDS = 77
# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id).to(device)
tokenizer = ldm.tokenizer
g_cpu = torch.Generator().manual_seed(888)
prompts = ["A painting of a squirrel eating a burger"]

controller = AttentionStore()

image, x_t,results = run_and_display(prompts, controller, run_baseline=False, generator=g_cpu)
results.save("cross_attention.png")

## base cross attention
show_cross_attention(controller, res=16, from_where=["up", "down"]) ### 单独up或者down有区别吗？

## replace
prompts = ["A painting of a squirrel eating a burger",
           "A painting of a lion eating a burger",
           "A painting of a cat eating a burger",
           "A painting of a deer eating a burger",
          ]
controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=.2) # cross会让self的step
_,_,results = run_and_display(prompts, controller, latent=x_t, run_baseline=True)
results.save("attentionreplace.png")


