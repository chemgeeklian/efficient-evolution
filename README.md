# Efficient evolution from general protein language models

This repository contains an updated implementation of the analysis described in the paper ["Efficient evolution of human antibodies from general protein language models"](https://www.nature.com/articles/s41587-023-01763-2).

The original scripts have been adapted to use modern ESM2 models and provide a more flexible framework for recommending mutations from custom model ensembles.

### Key Changes

*   **ESM2 Integration**: The core model backend has been updated from legacy ESM1 to modern ESM2 models via the Hugging Face `transformers` library.
*   **Flexible Model Configuration**: All models are now managed in `config/models.yaml`, allowing users to easily add new ESM models from Hugging Face or local fine-tuned checkpoints.
*   **Model Set Ensembles**: The script now uses "model sets" defined in the config file to generate mutation recommendations from a consensus of multiple models (e.g., an ensemble of original and fine-tuned models).

### Setup

1.  Clone this repository.
2.  Install the package and its dependencies:
    ```bash
    pip install .
    ```

### Usage

To get mutation recommendations for a protein sequence, run the `recommend.py` script. You must specify a sequence and a model set to use.

```bash
python -m efficient_evolution.recommend [SEQUENCE] --model-set [MODEL_SET_NAME]
```

**Arguments:**
*   `[SEQUENCE]`: The wildtype protein sequence you want to evolve.
*   `[MODEL_SET_NAME]`: The name of the model ensemble defined in `config/models.yaml` (e.g., `original_esm`, `esm2_finetuned`).

**Example:**

```bash
# Get recommendations using the fine-tuned ESM2 models
python -m efficient_evolution.recommend "MQWQTKLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEQSIPAIVDAYGDKALIGAGTVLKPEQVDALARMGCQLIVTPNIHSEVIRRAVGYGMTVCPGCATATEAFTALEAGAQALKIFPSSAFGPQYIKALKAVLPSDIAVFAVGGVTPENLAQWIDAGCAGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVQ" --model-set esm2_finetuned
```

or without installation, you can 

```bash
cd efficient-evolution
python efficient_evolution/recommend.py MQWQTNLPLIAILRGITPDEALAHVGAVIDAGFDAVEIPLNSPQWEKSIPQVVDAYGEQALIGAGTVLQPEQVDRLAAMGCRLIVTPNIQPEVIRRAVGYGMTVCPGCATASEAFSALDAGAQALKIFPSSAFGPDYIKALKAVLPPEVPVFAVGGVTPENLAQWINAGCVGAGLGSDLYRAGQSVERTAQQAAAFVKAYREAVK
```

The script will print a list of suggested mutations and their consensus count. Detailed results are saved to a new timestamped directory inside the `output/` folder.

### Configuration

You can add your own models or create new ensembles by editing `config/models.yaml`:

1.  **To add a new model**: Add an entry under the `models:` section with a unique name and its path (either a Hugging Face model ID or a local path to a checkpoint).
2.  **To create a new ensemble**: Add a new list under the `model_sets:` section, referencing the names of the models you defined.