
## 一、IPadapter原理
https://zhuanlan.zhihu.com/p/672366693
新一代“垫图”神器，IP-Adapter的完整应用解读|京东云技术团队 - 知乎 (zhihu.com)

A、将图片单独提出作为一种提示特征，通过解耦交叉注意力模块区分文本特征和图像特征。
B、本质为txt2img流程，prompt最为关键，利用IP-adapter强化参考图的作用。

C、两层controlnet，一层用来提供IP-adapter一层利用canny对需要添加的商品进行绘制、固化。


## 二、实现
1、image encoder对图像进行编码，训练过程当中image encoder参数冻结。
训练投影网络将图像embedding映射到长度为N=4的序列，图像和文本的信息维度相同。


text_embedding:[1,77,768]
image_embedding:[1,1024]经过proj network大小为[1,4,768]



2、cross attention部分

文本和图像信息使用相同的query，在原始UNet的每个cross-attention层之后添加，并且文本和图像的维度相同。
假设hidden_states = 320,cross attention输入维度unet.config.cross_attention_dim=768

cross attention 操作维度变化
Query:[1,4096,320],[b,seq_len,dim]
text:[1,77,768]-[1,77,320] 作为key和value
cross_attention:[1,4096,320]


Query:[1,4096,320]
ip_img:[1,4,768]-[1,4,320] 作为key和value
ip_cross_attention:[1,4096,320]

cross attention结果与原始query形状相同。


3、整个IP-adapter模型包含映射网络以及cross attention层参数




二、代码



1、模型加载
包括原始SD的模型以及image_encoder的加载
￼![Alt text](<assets/extras/Pasted Graphic 1.png>)

2、梯度冻结
￼![Alt text](<assets/extras/Pasted Graphic 2.png>)

3、映射层定义

clip_embeddings_dim映射为clip_extra_context_tokens*cross_attention_dim 

输入：image_embeds：shape()


输出：shape(-1,clip_extra_context_tokens,cross_attention_dim)
￼
![Alt text](<assets/extras/Pasted Graphic 3.png>)

4、模块初始化，自注意力层仍然为AttnProcessor()，但是cross attention层之后IPAttnProcessor()。此处IPAttnProcessor()同时实现text和img的cross attention。
￼
![Alt text](<assets/extras/Pasted Graphic 4.png>)
由于text和image共用query，因此to_q不需要训练。




AttnProcessor()以及IPAttnProcessor()定义在/mnt/workspace/workgroup_share/lhn/IP-Adapter/ip_adapter/attention_processor.py当中。


diffusers当中AttnProcessor的定义：
https://github.com/huggingface/diffusers/blob/a17d6d685870bd5b2b20d9c16498994fc945c7fd/src/diffusers/models/attention_processor.py#L729






5、IPadapter定义

ipadapter类，在/mnt/workspace/workgroup_share/lhn/IP-Adapter/ip_adapter/ip_adapter.py当中


init_proj函数：返回ImageProjModel实例image_proj_model，对图像维度进行映射。


get_image_embeds函数：首先使用clip_image_processor进行图像预处理操作，然后image_encoder得到clip_image_embeds，再经过映射函数得到image_prompt_embeds，最终返回image_prompt_embeds, uncond_image_prompt_embeds


## 与train_sdxl代码对比
### parse_args
#### noise_offset

理论：https://www.crosslabs.org//blog/diffusion-with-offset-noise
作用：改变噪声分布，影响整体明暗色调变化。

### model loading

sdxl具有两个text encoder。
```
tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
```

### data loading





### others
1、SDXL的vae使用fp32
















## 三、diffusers代码学习

1、自注意力替换
https://zhuanlan.zhihu.com/p/680035048
Stable Diffusion 中的自注意力替换技术与 Diffusers 实现 - 知乎 (zhihu.com)

U-Net 类的 attn_processors 属性会返回一个词典，它的 key 是每个处理类所在位置，比如 down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor，它的 value 是每个处理类的实例。

构造格式相同的词典，进行替换。
attn_processor_dict = {}
for k in unet.attn_processors.keys():
    if we_want_to_modify(k):
        attn_processor_dict[k] = MyAttnProcessor()
    else:
        attn_processor_dict[k] = AttnProcessor()

unet.set_attn_processor(attn_processor_dict)


## 四、存在问题

1、CNAttnProcessor：只使用文本是什么含义？

2、以attn1.processor结尾为self attention,其余的为cross attention，再看一下UNet的结构。
cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim



















