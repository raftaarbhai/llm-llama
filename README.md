# llm-llama



Plugin for [LLM](https://llm.datasette.io/) adding support for the Llama2 language model.

This plugin uses Llama2. The code was inspired by [abacaj/mpt-30B-inference](https://github.com/abacaj/mpt-30B-inference) and llm-mpt30b.

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-llama
```
After installing the plugin you will need to download the ~19GB model file. You can do this by running:

```bash
llm llama download
```

## Usage

This plugin adds a model called `llama` along side [some prompt engineering helpers](https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf). You can execute it like this:

```bash
llm -m llama "Three great names for a pet goat" -t weighting
```

You can pass the option `-t weighting` to apply prompt weighting. Full list of prompt techinques to be implementedinclude:

- [ ] Alternating
- [ ] Editing
- [ ] Weighting
- [ ] Blending
- [ ] Fusion

 

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

    cd llm-llama
    python3 -m venv venv
    source venv/bin/activate

Now install the dependencies and test dependencies:

    pip install -e '.[test]'

To run the tests:

    pytest
