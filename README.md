# Assignment 1 - OCR for English/Thai

## Setup

### Step 1

Use `generate_test_train_split.py` to generate an adequate train-test split as needed to train and validate your model.

```
usage: Train-Test-Split Generator [-h] [-l LANGUAGES [LANGUAGES ...]] [--train_set_resolutions TRAIN_SET_RESOLUTIONS [TRAIN_SET_RESOLUTIONS ...]]
                                  [--train_set_fontstyles TRAIN_SET_FONTSTYLES [TRAIN_SET_FONTSTYLES ...]] [--test_set_resolutions TEST_SET_RESOLUTIONS [TEST_SET_RESOLUTIONS ...]]
                                  [--test_set_fontstyles TEST_SET_FONTSTYLES [TEST_SET_FONTSTYLES ...]] [--test_set_size TEST_SET_SIZE] [-iv] [--inference] [-d] [-c CONFIG]

This program generates a train-test-split of the ThaiOCR dataset.

optional arguments:
  -h, --help            show this help message and exit
  -l LANGUAGES [LANGUAGES ...], --languages LANGUAGES [LANGUAGES ...]
                        Specify which language(s) should be included in the generated dataset.
  --train_set_resolutions TRAIN_SET_RESOLUTIONS [TRAIN_SET_RESOLUTIONS ...]
                        Specify which resolution(s) should be included in the generated train set, e.g. "200" or "200,300".
  --train_set_fontstyles TRAIN_SET_FONTSTYLES [TRAIN_SET_FONTSTYLES ...]
                        Specify which fontstyle(s) should be included in the generated test set, e.g. "italic" or "italic,bold".
  --test_set_resolutions TEST_SET_RESOLUTIONS [TEST_SET_RESOLUTIONS ...]
                        Specify which resolution(s) should be included in the generated test set, e.g. "200" or "200,300".
  --test_set_fontstyles TEST_SET_FONTSTYLES [TEST_SET_FONTSTYLES ...]
                        Specify which fontstyle(s) should be included in the generated test set, e.g. "italic" or "italic,bold".
  --test_set_size TEST_SET_SIZE
                        How large of a percentage must the test set be?
  -iv, --include_val    Include a validation set in the generated dataset.
  --inference           Generate a dataset for inference.
  -d, --debug_mode
  -c CONFIG, --config CONFIG
                        Specify the path to the config file.

Enjoy the program! :)
```

### Step 2

Train the model

