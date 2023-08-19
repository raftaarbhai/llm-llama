import llm
import random
import torch
import transformers
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import field_validator, Field
from typing import Optional



@llm.hookimpl
def register_models(register):
    register(Llama())

@llm.hookimpl
def register_commands(cli):
    @cli.group(name="llama")
    def llama_():
        "Commands for working with llama"

    @llama_.command()
    def download():
        "Download the <size>GB LLama-30B model file"
	# TODO: Replace with download for Llama file and appropriate warning about non-commercial use
        hf_hub_download(
            repo_id="TheBloke/Llama-2-7B-GGML",
            filename="llama-2-7b.ggmlv3.q2_K.bin",
        )


def generate(transitions, length, start_word=None):
    all_words = list(transitions.keys())
    next_word = start_word or random.choice(all_words)
    for i in range(length):
        yield next_word
        options = transitions.get(next_word) or all_words
        next_word = random.choice(options)

class Llama(llm.Model):
    model_id = "llama"
    class Options(llm.Options):
        technique: Optional[str] = Field(
            description="technique to use: alternating,weighting,weighted-average,blending,fusion",
            default=None
        )


    def execute(self, prompt, stream, response, conversation):
       model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
       tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
       if prompt.options.technique == "alternating":
           return self.execute_prompt_alternating(prompt.prompt, model, tokenizer)
       elif prompt.options.technique == "prompt-weighting":
           return self.execute_prompt_weighting(prompt.prompt, model, tokenizer)
      




    def execute_prompt_alternating(self, prompt: str, model, tokenizer):

       num_tokens = 20
       prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
       output_tokens = prompt_tokens.clone()
       alternate_prompts = ["blue", "red", "yellow"]
       insert_position=3
       for _ in range(num_tokens):
           alternate_index = len(output_tokens[0]) % len(alternate_prompts)
           alternate = alternate_prompts[alternate_index]

           alternate_tokens = tokenizer.encode(alternate, return_tensors="pt")

           input_ids = torch.cat((prompt_tokens[:, :insert_position], alternate_tokens, output_tokens[:, insert_position:]), dim=-1)
           next_token = model.generate(input_ids, max_length=input_ids.shape[1] + 1, do_sample = True)[:, -1].unsqueeze(0)

           output_tokens = torch.cat((output_tokens, next_token), dim=-1)
       generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
       return generated_text

    def execute_prompt_weighting(self, prompt, model, tokenizer, **kwargs):
        def modify_attention_mask(prompt, model, tokenizer):
            print("starting attention mask")
            tokens = []
            attention_modifiers = []
            add_space = False
            for token in re.split(r'\(|\)', prompt):
                print(f"current token is {token}")
                if ':' in token:
                    word, modifier = token.split(':')
                    modifier = float(modifier.strip())
                else:
                    word = token.strip()
                    modifier = 1.0
                current_tokens = tokenizer.tokenize(word)
                if add_space and current_tokens:
                    tokens.append('Ġ')  # Space token for GPT-2
                    attention_modifiers.append(1.0)
                tokens.extend(current_tokens)
                attention_modifiers.extend([modifier] * len(current_tokens))
                add_space = True
            attention_mask = torch.tensor([attention_modifiers])
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
            return input_ids, attention_mask
        input_ids, attention_mask = modify_attention_mask(prompt, model, tokenizer)
        print(attention_mask)
        # Set the modified attention mask
        model.config.attention_probs_dropout_prob = 0.0
        with torch.no_grad():
            output_sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            print(output_sequences)
        return tokenizer.decode(output_sequences[0], skip_special_tokens=True)
