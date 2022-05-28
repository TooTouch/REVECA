import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import GPT2Model, GPT2Config
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from timm import create_model as timm_create_model
from timm.models.layers import PatchEmbed
from utils import accuracy
import loralib as lora

import math
import logging

_logger = logging.getLogger('train')

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)



class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

        

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out

class MultiModalDecoder(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self, 
        hidden_states, 
        attention_mask, 
        encoder_hidden_states,
        use_cache=None
   ):   

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        output_shape = hidden_states.size()
        batch_size = hidden_states.shape[0]

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=hidden_states.device)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        # blocks
        hidden_states = self.drop(hidden_states)
        presents = () if use_cache else None
        for i, block in enumerate(self.h):
            outputs = block(
                hidden_states,
                attention_mask         = attention_mask,
                encoder_hidden_states  = encoder_hidden_states,
                encoder_attention_mask = encoder_attention_mask,
                use_cache              = use_cache,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
    
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states
        )


class VideoBoudnaryCoCa(nn.Module, GenerationMixin):
    def __init__(
        self, 
        num_tokens, 
        image_encoder, 
        unimodal_decoder, 
        multimodal_decoder,
        aggregation_frames_method,
        caption_loss_weight,
        contrastive_loss_weight,
        num_img_queries = 256,
        use_frame_position = False,
        use_seg_features = False,
        use_tsn_features = False,
        use_temporal_pairwise_difference = False,
        num_frames = 21,
        heads = 8,
        pad_id = None,
        device = 'cpu'
    ):
        super().__init__()
        GenerationMixin.__init__(self)

        self.device = device

        # config
        self.config = GPT2Config()
        self.config.is_encoder_decoder = True
        self.main_input_name = 'frames'

        # models
        self.image_encoder = image_encoder
        self.unimodal_decoder = unimodal_decoder 
        self.multimodal_decoder = multimodal_decoder 

        # aggregation frame method
        self.aggregation_frames_method = aggregation_frames_method

        freeze_model_and_make_eval_(self.image_encoder)

        # loss weights
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # pad id
        self.pad_id = pad_id

        # attention pooling for image tokens
        self.num_img_queries = num_img_queries
        self.unimodal_decoder_dim = self.unimodal_decoder.config.n_embd
        self.image_dim = self.image_encoder.embed_dim 
        # image_dim = image_dim if not self.use_frame_position else image_dim + 1 #TODO: 만약 frame_pos_embed가 잘되면 삭제
        dim_head = self.image_dim // heads
        self.img_queries = nn.Parameter(torch.randn(self.num_img_queries + 1, self.unimodal_decoder_dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(
            dim          = self.unimodal_decoder_dim, 
            context_dim  = self.image_dim, 
            dim_head     = dim_head, 
            heads        = heads, 
            norm_context = True
        )

        self.img_attn_pool_norm = LayerNorm(self.unimodal_decoder_dim)
        self.cls_norm = LayerNorm(self.unimodal_decoder_dim)

        # seg features
        self.use_seg_features = use_seg_features
        if self.use_seg_features:
            self.seg_patch_embed = PatchEmbed(
                img_size   = self.image_encoder.patch_embed.img_size, 
                patch_size = self.image_encoder.patch_embed.patch_size, 
                in_chans   = 1,
                embed_dim  = self.image_encoder.patch_embed.proj.out_channels
            )

        # tsn features
        self.use_tsn_features = use_tsn_features
        if self.use_tsn_features:
            self.tsn_embed = nn.Linear(2048, self.unimodal_decoder_dim)
            self.tsn_pos_embed = nn.Embedding(3, self.unimodal_decoder_dim)

        # Temporal Pairwise Difference
        self.use_temporal_pairwise_difference = use_temporal_pairwise_difference

        # use frame position
        self.use_frame_position = use_frame_position
        self.num_frames = num_frames
        if self.use_frame_position:
            self.frame_pos_embed = nn.Embedding(self.num_frames, self.image_dim)

        # contrastive learning temperature
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # to logits
        self.to_logits = nn.Sequential(
            LayerNorm(self.unimodal_decoder_dim),
            nn.Linear(self.unimodal_decoder_dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = nn.Parameter(self.unimodal_decoder.wte.weight[:-1,:])

    def forward(self, captions, frames=None, seg_features=None, tsn_features=None, frames_embed=None, labels=None, return_loss=False, **kwargs):

        # unimodal decoding
        captions_cls_embed, captions_embed = self.embed_captions(captions, cls_return=return_loss)

        # image encoding
        if frames_embed is None:
            captions['attention_mask'] = captions['attention_mask'][:,:-1]
            frames_cls_embed, frames_embed = self.aggregation_frames(
                frames       = frames, 
                seg_features = seg_features,
                tsn_features = tsn_features
            )
        else:
            captions['attention_mask'] = None

        # multimodal decoding
        output = self.multimodal_decoder(
            hidden_states         = captions_embed,
            attention_mask        = captions['attention_mask'],
            encoder_hidden_states = frames_embed
        )

        logits = self.to_logits(output['last_hidden_state'])

        if return_loss:
            acc1, acc5 = accuracy(logits, labels)
            caption_loss, contrastive_loss = self.calc_loss(captions_cls_embed, frames_cls_embed, logits, labels)
            return acc1, acc5, caption_loss, contrastive_loss

        else:
            return CausalLMOutputWithCrossAttentions(
                logits = logits
            )


    def embed_captions(self, captions, cls_return=True):
        unimodal_output = self.unimodal_decoder(**captions)
        
        if cls_return:
            cls_embed = unimodal_output['last_hidden_state'][:,-1]
            captions_embed = unimodal_output['last_hidden_state'][:,:-1]

            # get text cls token
            cls_embed = self.cls_norm(cls_embed)
            return cls_embed, captions_embed
        else:
            return None, unimodal_output['last_hidden_state']


    def embed_frame(self, frame, seg_features=None, pos_idx=0):    
        self.image_encoder.eval()
        with torch.no_grad():
            image_embed = self.image_encoder.forward_features(frame).detach()

        # seg features
        if seg_features is not None:
            seg_embed = self.seg_patch_embed(seg_features)
            image_embed[:,1:,:] += seg_embed

        # use frame position
        if self.use_frame_position:
            image_embed = self.add_frame_position(image_embed, pos_idx)

        # attention pool image tokens
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_embed.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_embed)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]


    def aggregation_frames(self, frames, seg_features=None, tsn_features=None):
        """
        cls_embed: (batch, dim)
        frames_embed: (batch, n_query, dim)
        """
        num_half_frames = self.num_frames//2
        batch_size = frames['boundary'].size(0)
        device = frames['boundary'].device

        all_cls_embed = torch.zeros(
            (batch_size, self.num_frames, self.unimodal_decoder.config.n_embd)).to(device) 
        all_frames_embed = torch.zeros(
            (batch_size, self.num_frames, self.num_img_queries, self.unimodal_decoder.config.n_embd)).to(device)

        pos_idx = 0

        # before boundary
        for i in range(frames['before'].shape[1]):
            cls_embed, frames_embed = self.embed_frame(
                frame        = frames['before'][:,i,:,:], 
                seg_features = seg_features['before'][:,i,:,:] if seg_features is not None else seg_features,
                pos_idx      = pos_idx
            )
            all_cls_embed[:,pos_idx,...] = cls_embed
            all_frames_embed[:,pos_idx,...] = frames_embed
            pos_idx += 1

        # boundary
        cls_embed, frames_embed = self.embed_frame(
            frame        = frames['boundary'], 
            seg_features = seg_features['boundary'] if seg_features is not None else seg_features,
            pos_idx      = pos_idx
        )
        all_cls_embed[:,pos_idx,...] = cls_embed
        all_frames_embed[:,pos_idx,...] = frames_embed
        pos_idx += 1

        # before boundary
        for i in range(frames['after'].shape[1]):
            cls_embed, frames_embed = self.embed_frame(
                frame        = frames['after'][:,i,:,:], 
                seg_features = seg_features['after'][:,i,:,:] if seg_features is not None else seg_features,
                pos_idx      = pos_idx
            )
            all_cls_embed[:,pos_idx,...] = cls_embed
            all_frames_embed[:,pos_idx,...] = frames_embed
            pos_idx += 1

        # aggregation
        cls_embed = all_cls_embed.mean(dim=1)
        frames_embed = getattr(self, self.aggregation_frames_method)(all_frames_embed)

        # Temperal Pairwise Difference
        if self.use_temporal_pairwise_difference:
            tpd_cls_embed = self.temporal_pairwise_difference(all_cls_embed)
            frames_embed = torch.cat([frames_embed, tpd_cls_embed], dim=1)

        # TSN features
        if self.use_tsn_features:
            tsn_embed = self.embed_tsn_features(tsn_features)
            frames_embed = torch.cat([frames_embed, tsn_embed], dim=1)

        return cls_embed, frames_embed                

    def aggregation_frames_method1(self, all_frames_embed):
        frames_embed = torch.stack(all_frames_embed).mean(dim=1) 
        
        return frames_embed

    def aggregation_frames_method2(self, all_frames_embed):   
        frames_embed = rearrange(all_frames_embed, 'b f n d -> b (f n) d')
        
        return frames_embed

    def add_frame_position(self, image_tokens, pos_idx=0):
        batch_size, num_tokens, _ = image_tokens.size()
        device = image_tokens.device
        # device = image_tokens.device 

        # pos = torch.zeros((batch_size, num_tokens, 1)).to(self.device)
        # pos -= 2 * (1 - pos_idx/(self.num_frames -1)) - 1
        # image_tokens = torch.cat([image_tokens, pos], dim=-1)

        frame_pos_embed = self.frame_pos_embed(torch.tensor(pos_idx).to(device))
        frame_pos_embed = repeat(frame_pos_embed, 'd -> b t d', b=batch_size, t=num_tokens)

        image_tokens += frame_pos_embed

        return image_tokens

    def temporal_pairwise_difference(self, all_cls_embed):
        device = all_cls_embed

        before_cls = all_cls_embed[:,:10]
        boundary_cls = all_cls_embed[:,10]
        after_cls = all_cls_embed[:,-10:]
        
        num_half_frames = self.num_frames//2
        batch_size = boundary_cls.size(0)

        tpd_cls_embed = torch.zeros((batch_size, num_half_frames ** 2 + num_half_frames * 2,boundary_cls.size(1))).to(device)
        for b_idx in range(num_half_frames):
            for a_idx in range(num_half_frames):
                tpd_cls_embed[:,(b_idx * num_half_frames) + (a_idx % num_half_frames),:] = before_cls[:,b_idx,:] - after_cls[:,a_idx,:]
        
        for b_idx in range(num_half_frames):
            tpd_cls_embed[:,num_half_frames ** 2 + b_idx,:] = before_cls[:,b_idx,:] - boundary_cls
        
        for a_idx in range(num_half_frames):
            tpd_cls_embed[:,num_half_frames ** 2 + num_half_frames + a_idx,:] = boundary_cls - after_cls[:,a_idx,:]

        return tpd_cls_embed


    def embed_tsn_features(self, tsn_features):
        device = tsn_features['before'].device
        batch_size = tsn_features['before'].size(0)
        tsn_embed = torch.zeros((batch_size, 3, self.unimodal_decoder_dim)).to(device)

        # embedding
        tsn_embed[:,0,:] = self.tsn_embed(tsn_features['before'])
        tsn_embed[:,1,:] = self.tsn_embed(tsn_features['after'])
        tsn_embed[:,2,:] = self.tsn_embed(tsn_features['before'] - tsn_features['after'])

        # add position embedding using broadcasting
        # tsn_embed[i] = (batch size x dim) + (1 x dim)
        tsn_embed[:,0,:] += self.tsn_pos_embed(torch.tensor(0).to(device)) 
        tsn_embed[:,1,:] += self.tsn_pos_embed(torch.tensor(1).to(device))
        tsn_embed[:,2,:] += self.tsn_pos_embed(torch.tensor(2).to(device))

        return tsn_embed



    def calc_loss(self, captions_cls_embed, frame_cls_embed, pred_captions, true_captions):
        batch, device = captions_cls_embed.shape[0], captions_cls_embed.device

        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)
        pred_captions = rearrange(pred_captions, 'b n c -> b c n')
        caption_loss = ce(pred_captions, true_captions)
        caption_loss = caption_loss * self.caption_loss_weight

        # calculate contrastive loss
        sim = einsum('i d, j d -> i j', captions_cls_embed, frame_cls_embed)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss, contrastive_loss


    def get_encoder(self):
        return self.aggregation_frames

    def _prepare_model_inputs(
        self,
        inputs = None,
        bos_token_id = None,
        model_kwargs = None,
    ):

        return inputs['frames']['boundary'], inputs, model_kwargs

    def prepare_inputs_for_generation(self, captions, **model_kwargs):
        captions = captions.to(self.device)
        return {'captions':{'input_ids':captions.to(self.device)}, 'frames_embed':model_kwargs['encoder_outputs']}

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name = None
    ):
        frames = model_input_name # actually this is frames
        encoder = self.get_encoder()

        model_kwargs["encoder_outputs"] = encoder(**frames)[1]

        return model_kwargs


    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size = 1,
        is_encoder_decoder = False,
        encoder_outputs = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            encoder_outputs = encoder_outputs.index_select(
                0, expanded_return_idx.to(encoder_outputs.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
        
class Conv1D_LoRA(Conv1D, lora.LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        Conv1D.__init__(self, out_channels, in_channels, **kwargs)
        lora.LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_channels))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels, r))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        Conv1D.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        Conv1D.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            size_out = x.size()[:-1] + (self.nf,)
            result = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            result = result.view(size_out)
            
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)



def set_lora(args, model):
    for layer in model.h:
        layer.attn.c_attn = Conv1D_LoRA(
            model.config.n_embd, model.config.n_embd * 3, 
            r               =   args.lora_r, 
            lora_alpha      =   args.lora_alpha, 
            lora_dropout    =   args.lora_dropout, 
            merge_weights   =   args.merge_weights,
        )
        
        if model.config.add_cross_attention:
            layer.crossattention.c_attn = Conv1D_LoRA(
                model.config.n_embd, model.config.n_embd * 2, 
                r               =   args.lora_r, 
                lora_alpha      =   args.lora_alpha, 
                lora_dropout    =   args.lora_dropout, 
                merge_weights   =   args.merge_weights,
            )

            layer.crossattention.q_attn = Conv1D_LoRA(
                model.config.n_embd, model.config.n_embd, 
                r               =   args.lora_r, 
                lora_alpha      =   args.lora_alpha, 
                lora_dropout    =   args.lora_dropout, 
                merge_weights   =   args.merge_weights,
            )  
        
    return model

def calc_params(model):
    param = 0
    for p in model.parameters():
        if p.requires_grad:
            param += p.numel()

    return param

def create_model(args, tokenizer):
    # image encoder
    image_encoder = timm_create_model(args.image_modelname, pretrained=True, img_size=args.img_size)

    # unimodal decoder
    if args.use_lora:
        config = GPT2Config.from_pretrained(args.unimodal_modelname)
        unimodal_decoder = GPT2Model(config=config)
        unimodal_decoder = set_lora(args, unimodal_decoder)
        unimodal_decoder.load_state_dict(
            GPT2Model.from_pretrained(args.unimodal_modelname, config=config).state_dict(), 
            strict=False
        )

        _logger.info('Build a Unimodal Decoder with LoRA')
        before_param = calc_params(unimodal_decoder)

        # set lora grad
        lora.mark_only_lora_as_trainable(unimodal_decoder)

        after_param = calc_params(unimodal_decoder)
        _logger.info('Trainable parameters of a Unimodal Decoder change {} to {}'.format(before_param, after_param))

        
    else:
        unimodal_decoder = GPT2Model.from_pretrained(args.unimodal_modelname)
    unimodal_decoder.resize_token_embeddings(len(tokenizer))

    # multimodal decoder
    config = GPT2Config.from_pretrained(args.multimodal_modelname)
    config.add_cross_attention = True
    if args.use_lora:
        multimodal_decoder = MultiModalDecoder(config=config)
        multimodal_decoder = set_lora(args, multimodal_decoder)
        multimodal_decoder.load_state_dict(
            GPT2Model.from_pretrained(args.multimodal_modelname, config=config).state_dict(), 
            strict=False
        )

        _logger.info('Build a Unimodal Decoder with LoRA')
        before_param = calc_params(multimodal_decoder)

        # set lora grad
        lora.mark_only_lora_as_trainable(multimodal_decoder)

        after_param = calc_params(unimodal_decoder)
        _logger.info('Trainable parameters of a Multimodal Decoder change {} to {}'.format(before_param, after_param))

    else:
        multimodal_decoder = MultiModalDecoder.from_pretrained(args.multimodal_modelname, config=config)


    model = VideoBoudnaryCoCa(
        num_tokens                       = len(tokenizer)-1, 
        image_encoder                    = image_encoder, 
        unimodal_decoder                 = unimodal_decoder, 
        multimodal_decoder               = multimodal_decoder,
        aggregation_frames_method        = args.aggregation_frames_method,
        caption_loss_weight              = args.caption_loss_weight,
        contrastive_loss_weight          = args.contrastive_loss_weight,
        num_img_queries                  = args.num_img_queries,
        use_frame_position               = args.use_frame_position,
        use_seg_features                 = args.use_seg_features,
        use_tsn_features                 = args.use_tsn_features,
        use_temporal_pairwise_difference = args.use_temporal_pairwise_difference,
        num_frames                       = 2 * args.max_sample_num + 1,
        heads                            = args.num_heads,
        pad_id                           = tokenizer.encode(tokenizer.eos_token)[0],
        device                           = args.device
    )

    return model
