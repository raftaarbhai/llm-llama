import llm
import random

@llm.hookimpl
def register_models(register):
    register(Llama())

@llm.hookimpl
def register_commands(cli):
    @cli.group(name="llama")
    def llama_():
        "Commands for working with llama"

    @mpt30b_.command()
    def download():
        "Download the <size>GB LLama-30B model file"
	# TODO: Replace with download for Llama file and appropriate warning about non-commercial use
        #hf_hub_download(
        #    repo_id="TheBloke/mpt-30B-chat-GGML",
        #    filename="mpt-30b-chat.ggmlv0.q4_1.bin",
        #)

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
        technique: str = None

    def execute(self, prompt, stream, response, conversation):
        text = prompt.prompt
        transitions = build_markov_table(text)
        for word in generate(transitions, 20):
            yield word + ' '
