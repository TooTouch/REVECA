import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


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
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)


        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
    
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states
        )



class VideoBoudnaryCoca(nn.Module):
    def __init__(
        self, 
        num_tokens, 
        image_encoder, 
        unimodal_decoder, 
        multimodal_decoder,
        caption_loss_weight,
        contrastive_loss_weight,
        num_img_queries = 256,
        heads = 8,
        pad_id = None
    ):

        # models
        self.image_encoder = image_encoder
        self.unimodal_decoder = unimodal_decoder 
        self.multimodal_decoder = multimodal_decoder 

        # loss weights
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # pad id
        self.pad_id = pad_id

        # cls embedding
        self.text_cls_token = nn.Parameter(torch.randn(dim))

        # attention pooling for image tokens
        dim = self.unimodal_decoder.config.n_embd
        image_dim = self.image_encoder.embed_dim
        dim_head = image_dim // heads
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)

        self.img_attn_pool_norm = LayerNorm(dim)
        self.cls_norm = LayerNorm(dim)

        # contrastive learning temperature
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # to logits
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.unimodal_decoder.wte.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, captions, frames, return_loss=False):
        # split captions and labels
        if return_loss:
            captions, labels = captions[:, :-1], captions[:, 1:]

        # unimodal decoding
        captions_cls_embed, captions_embed = self.embed_captions(captions)

        # image encoding
        frames_cls_embed, frames_embed = self.aggregation_frames(frames)

        # multimodal decoding
        output = self.multimodal_decoder(
            hidden_states = captions_embed,
            attention_mask = captions['attention_mask'],
            encoder_hidden_states = frames_embed
        )

        logits = self.to_logits(output['last_hidden_state'])

        if return_loss:
            return self.calc_loss(captions_cls_embed, frames_cls_embed, logits, labels)


    def embed_captions(self, captions):
        batch = captions['input_ids'].shape[0]
    
        cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        captions['input_ids'] = torch.cat((captions['input_ids'], cls_tokens), dim=-2)
        
        cls_attention_mask = repeat(torch.tensor([1]),'1 -> b 1', b=batch)
        captions['attention_mask'] = torch.cat((captions['attention_mask'], cls_attention_mask), dim=-1)

        unimodal_output = self.unimodal_decoder(**captions)
        cls_embed = unimodal_output['last_hidden_state'][:,-1]
        captions_embed = unimodal_output['last_hidden_state'][:,:-1]

        # get text cls token
        cls_embed = self.cls_norm(cls_embed)
        return cls_embed, captions_embed


    def embed_frame(self, frame):    
        self.img_encoder.eval()
        with torch.no_grad():
            image_tokens = self.img_encoder.forward_features(frame).detach()

        # attention pool image tokens
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]


    def aggregation_frames(self, frames):
        cls_embed_list = []
        frames_embed_list = []

        # boundary
        cls_embed, frames_embed = self.embed_frame(frames['boundary'])
        cls_embed_list.append(cls_embed)
        frames_embed_list.append(frames_embed)

        # before boundary
        for f in frames['before']:
            cls_embed, frames_embed = self.embed_frame(f)
            cls_embed_list.append(cls_embed)
            frames_embed_list.append(frames_embed)

        # before boundary
        for f in frames['after']:
            cls_embed, frames_embed = self.embed_frame(f)
            cls_embed_list.append(cls_embed)
            frames_embed_list.append(frames_embed)

        # aggregation
        cls_embed = torch.stack(cls_embed_list).mean(dim=0)
        frames_embed = torch.stack(frames_embed_list).mean(dim=0) 

        return cls_embed, frames_embed


    def calc_loss(self, captions_cls_embed, frame_cls_embed, pred_captions, true_captions):
        batch, device = captions_cls_embed.shape[0], captions_cls_embed.device

        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)
        pred_captions = rearrange(pred_captions, 'b n c -> b c n')
        caption_loss = ce(pred_captions, true_captions, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # calculate contrastive loss
        sim = einsum('i d, j d -> i j', captions_cls_embed, frame_cls_embed)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss