import torch
import torch.nn as nn

from collections import OrderedDict

from .encoders import AudioEncoder
from .encoders import TextEncoder



class PromptLearner(nn.Module):
    def __init__(self, args, text_encoder, pengi):
        super().__init__()


        self.args = args
        classnames = args.classnames
        n_cls = len(classnames)

        n_ctx = args.n_ctx # 16
        # ctx_dim = args.ctx_dim # 512

        token_emb_dim = text_encoder.base.embeddings.token_embedding.embedding_dim  # dimension of token embeddings (usually 512)
        feature_dim = pengi.args.d_proj  # dimension of audio features (usually 1024)

        #tokenizer = pengi.enc_tokenizer
        #classnames_token_lens = [len(tokenizer.encode(class_name)) - 2  for class_name in classnames]

        print("Initializing a Generic Context for COCOOP ...")
        ctx_vectors = torch.empty(n_ctx, token_emb_dim)
        torch.nn.init.normal_(ctx_vectors, std=0.02)
        ctx = torch.nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {class_name}." for class_name in classnames]

        tokenized_prompts = pengi.preprocess_text(prompts, enc_tok=True, add_text=False)
        # tokenized_prompts = {key:value.to(args.device) for key,value in tokenized_prompts.items()}  # move tensors to device

        with torch.no_grad():
            tokenized_prompts_embeddings = text_encoder.base.embeddings.token_embedding(tokenized_prompts['input_ids'])   # [batch_size, seq_length, embed_dim]
        
        # breakpoint()
        sos_embedding = tokenized_prompts_embeddings[:,:1,:]
        classnames_and_pad_embedding = tokenized_prompts_embeddings[:,1+n_ctx:,:]

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts
        self.ctx = ctx
        self.register_buffer("sos", sos_embedding)  # SOS
        self.register_buffer("classes_pad", classnames_and_pad_embedding)  # CLASSNAMES, PADDING Token Embeddings


        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(feature_dim, feature_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(feature_dim // 16, token_emb_dim))
        ]))


    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, token_emb_dim)
        # prefix: the sos token, with shape of (n_cls, 1, token_emb_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, token_emb_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, audio_features):
        # audio_features: (batch, feature_dim)
        
        ctx = self.ctx.unsqueeze(0)        # (1, n_ctx, token_emb_dim)

        prefix = self.sos
        suffix = self.classes_pad

        # prompts_token_embeddings = torch.cat( [prefix, ctx, suffix ], dim=1)
        bias = self.meta_net(audio_features)  # (batch, token_emb_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, token_emb_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, token_emb_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts_token_embeddings = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, token_emb_dim)
            prompts_token_embeddings.append(pts_i)
        prompts_token_embeddings = torch.stack(prompts_token_embeddings)
        
        return self.tokenized_prompts['input_ids'].to(self.args.device), prompts_token_embeddings, self.tokenized_prompts['attention_mask'].to(self.args.device)


class CustomPENGI(nn.Module):
    def __init__(self,args,pengi):
        super().__init__()

        self.args = args
        pengi_args  = pengi.args
        self.pengi_args = pengi_args
        
        pengi_args.specaug = args.spec_aug

        self.process_text = pengi.preprocess_text

        self.audio_encoder = AudioEncoder(
                    pengi_args.audioenc_name, pengi_args.out_emb, pengi_args.d_proj,
                    pengi_args.sampling_rate, pengi_args.window_size, pengi_args.hop_size, pengi_args.mel_bins, pengi_args.fmin, pengi_args.fmax, pengi_args.classes_num, 
                    pengi_args.specaug, pengi_args.mixup, pengi_args.use_pretrained_audioencoder, pengi_args.freeze_audio_encoder_weights,
                    pengi_args.use_precomputed_melspec, pengi_args.pretrained_audioencoder_path)

        self.text_encoder = TextEncoder(
                    pengi_args.d_proj, 
                    pengi_args.text_model, pengi_args.transformer_embed_dim,
                    pengi_args.freeze_text_encoder_weights)        

        # load the weights of the pengi pre-trained audio and text encoders
        print("\n\nCOCOOP: loading the weights of the pengi pre-trained audio and text encoders ...\n\n")
        self.audio_encoder.load_state_dict(pengi.model.audio_encoder.state_dict())
        self.text_encoder.load_state_dict(pengi.model.caption_encoder.state_dict())

        self.audio_encoder.eval()
        self.text_encoder.eval()


        self.prompt_learner = PromptLearner(args, self.text_encoder, pengi)

        self.device = args.device

        
        

    def get_zeroshot_text_features(self):

        prompts = [f"{self.args.prompt_prefix} {class_name}." for class_name in self.args.classnames]

        tokenized_prompts = self.process_text(prompts, enc_tok=True, add_text=False)
        prompts_tokens = tokenized_prompts['input_ids'].to(self.device)
        prompts_token_embeddings = self.text_encoder.base.embeddings.token_embedding(prompts_tokens).to(self.device)   # [batch_size, seq_length, embed_dim]
        prompts_attention_mask = tokenized_prompts['attention_mask'].to(self.device)
        
        text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings, "attention_mask": prompts_attention_mask}
        
        with torch.no_grad():
            text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features


    def zeroshot_logits(self, audio_features):
        # audio_features: (batch, feature_dim)
        # text_features: (n_cls, feature_dim)

        if not hasattr(self, 'zeroshot_text_features'): self.zeroshot_text_features = self.get_zeroshot_text_features()
        
        logit_scale = 100.0
        logits = logit_scale * audio_features @ self.zeroshot_text_features.t()
        return logits
    
    def forward(self, audio):
        # breakpoint()
        # with torch.no_grad():
        audio_features = self.audio_encoder(audio)[0] # audio_features shape [n_audio_files, 1024]
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        # if self.args.eval_only:
        prompts_tokens, prompts_token_embeddings, prompts_attention_mask = self.prompt_learner(audio_features)
        # else:
        #     masked_audio_features = self.audio_encoder(mask_audio_spans(audio))[0]
        #     masked_audio_features = masked_audio_features / masked_audio_features.norm(dim=-1, keepdim=True)

        #     prompts_tokens, prompts_token_embeddings, prompts_attention_mask = self.prompt_learner(masked_audio_features)
        

        logit_scale = 100.0
        logits = []
        for prompts_token_embeddings_i, audio_features_i in zip(prompts_token_embeddings, audio_features):
            text = {"input_ids": prompts_tokens, "inputs_embeds": prompts_token_embeddings_i, "attention_mask": prompts_attention_mask}
            text_features = self.text_encoder(text) # text_features shape [n_text_prompts, 1024]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * audio_features_i @ text_features.t()
            logits.append(l_i)

        logits = torch.stack(logits)
        
        if self.args.use_zebra:
            logits = 0.5*logits + 0.5*self.zeroshot_logits(audio_features)
        
        return logits






# def mask_audio_spans(audio, mask_prob=0.15, min_span=16000, max_span=32000):
#     """
#     Mask random spans with variable lengths.
    
#     Args:
#         audio: [batch, num_samples]
#         mask_prob: Fraction of audio to mask
#         min_span: Minimum span length in samples
#         max_span: Maximum span length in samples
#     """
#     batch_size, num_samples = audio.shape
    
#     masked_audio = audio.clone()
#     mask = torch.zeros_like(audio, dtype=torch.bool)
    
#     for b in range(batch_size):
#         total_masked = 0
#         target_masked = int(num_samples * mask_prob)
        
#         while total_masked < target_masked:
#             # Random span length
#             span_length = torch.randint(min_span, max_span + 1, (1,)).item()
            
#             # Random start position
#             if num_samples - span_length <= 0:
#                 break
#             start = torch.randint(0, num_samples - span_length, (1,)).item()
#             end = start + span_length
            
#             # Apply mask
#             masked_audio[b, start:end] = 0
#             mask[b, start:end] = True
            
#             total_masked += span_length
    
#     return masked_audio


