import llm
import random
import torch
import transformers
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead
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

class Llama(llm.Model):
    model_id = "llama"
    class Options(llm.Options):
        technique: Optional[str] = Field(
            description="technique to use: alternating,weighting,weighted-average,blending,fusion",
            default=None
        )

    def execute(self, prompt, stream, response, conversation):

       if prompt.options.technique == "alternating":
           model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
           tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
           return self.execute_prompt_alternating(prompt.prompt, model, tokenizer)
       elif prompt.options.technique == "prompt-weighting":
           model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
           tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
           return self.execute_prompt_weighting(prompt.prompt, model, tokenizer)
       elif prompt.options.technique == "blending":
           tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
           model = AutoModelWithLMHead.from_pretrained('gpt2-xl', device_map='auto')
           return self.execute_prompt_blending(prompt.prompt, model, tokenizer)


    def execute_prompt_alternating(self, prompt: str, model, tokenizer):
       num_tokens = 20
       prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
       output_tokens = prompt_tokens.clone()

       # TODO: move alternate prompts to be an argument
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
                    tokens.append(' ')  # Space token
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

    def execute_prompt_blending(self, prompt, model, tokenizer):
        # TODO: Move sequences and weights to be an argument
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        # Get the embeddings for the entire prompt
        all_embeddings = model.transformer.wte(input_ids)
        # List of sequences to average
        sequences = ["delicious chow mein", "delicious ice cream", "tasty pizza"]
        # List of weights for each sequence
        weights = [0.6, 0.3, 0.1]
        assert len(sequences) == len(weights), "Weights and sequences must have the same length."
        # Tokenize and retrieve the embeddings for the sequences
        sequence_embeddings = []
        for seq in sequences:
            input_ids_seq = tokenizer.encode(seq, return_tensors='pt')
            embeddings_seq = model.transformer.wte(input_ids_seq)
            sequence_embeddings.append(embeddings_seq.mean(dim=1))
        # Calculate the weighted average embeddings for the desired sequences
        weights_tensor = torch.tensor(weights).view(-1, 1, 1).to(all_embeddings.device)
        weighted_embeddings = torch.stack(sequence_embeddings, dim=0) * weights_tensor
        average_embedding = weighted_embeddings.sum(dim=0)
        # TODO: Change this to input
        insert_position = 3
        # Concatenate the averaged embeddings with the prompt embeddings at the specified position
        modified_embeddings = torch.cat([all_embeddings[:, :insert_position], average_embedding.unsqueeze(1), all_embeddings[:, insert_position:]], dim=1)
        # Use the modified embeddings as input
        output = model.generate(inputs_embeds=modified_embeddings, do_sample=True, max_length=100)
        decoded_output = tokenizer.decode(output[0])

        return decoded_output
