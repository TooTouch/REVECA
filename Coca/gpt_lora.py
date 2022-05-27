from transformers import AutoModelForPreTraining, AutoTokenizer, AutoConfig
import loralib as lora
import torch
import torch.nn as nn


class GPT2LoRA(nn.Module) :
    def __init__(self, config) :
        super().__init__()
        self.model = AutoModelForPreTraining.from_pretrained("gpt2-large", config=config)
        self.config = config

        if True in config.enable_lora :
            print("Enable LORA")
            self.get_pretrained_state_dict()
            self.set_lora()
            self.apply(self.__init_weights)
            self.model.load_state_dict(self.pretrained_state_dict, strict=False)
            lora.mark_only_lora_as_trainable(self.model)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_pretrained_state_dict(self) :
        self.pretrained_state_dict = {}
        for layer_name, param in self.model.state_dict().items() :
            self.pretrained_state_dict[layer_name] = param

    def set_lora(self) :
        for layer in self.model.transformer.h :
            layer.attn.c_attn = lora.MergedLinear(
                self.config.n_embd, self.config.n_embd * 3, 
                r               =   self.config.lora_attn_dim, 
                lora_alpha      =   self.config.lora_attn_alpha, 
                lora_dropout    =   self.config.lora_dropout, 
                enable_lora     =   self.config.enable_lora, 
                fan_in_fan_out  =   self.config.fan_in_fan_out,
                merge_weights   =   self.config.merge_weights,
        )
    
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

def main() :
    import time
    config = AutoConfig.from_pretrained("gpt2-large")
    config.lora_attn_dim = 8
    config.lora_attn_alpha = 8
    config.lora_dropout = 0.1
    config.enable_lora = [True, False, True] #[True, False, True]
    config.fan_in_fan_out = True
    config.merge_weights = False
    config.freeze_pretrained_layers = True
    model = GPT2LoRA(config).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    for i in range(5) :
        print(i)    
        text = ["Replace me by any text you'd like."]*10
        encoded_input = tokenizer.batch_encode_plus(text, return_tensors='pt').to("cuda")
        output = model(**encoded_input)

    import torch
    result = tokenizer.batch_decode(torch.argmax(output.logits, dim  = 2))
    print(result)
    time.sleep(5)
if __name__ == "__main__" :
    main()