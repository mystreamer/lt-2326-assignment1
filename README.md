# Assignment 1 - OCR for English/Thai

## Setup

Make sure you have cd'd into this repository.

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
                        Specify which resolution(s) should be included in the generated train set, e.g. "200" or "200" "300" (space separated).
  --train_set_fontstyles TRAIN_SET_FONTSTYLES [TRAIN_SET_FONTSTYLES ...]
                        Specify which fontstyle(s) should be included in the generated test set, e.g. "italic" or "italic" "bold" (space-separated).
  --test_set_resolutions TEST_SET_RESOLUTIONS [TEST_SET_RESOLUTIONS ...]
                        Specify which resolution(s) should be included in the generated test set, e.g. "200" or "200" "300" (space-separated).
  --test_set_fontstyles TEST_SET_FONTSTYLES [TEST_SET_FONTSTYLES ...]
                        Specify which fontstyle(s) should be included in the generated test set, e.g. "italic" or "italic" "bold" (space-separated).
  --test_set_size TEST_SET_SIZE
                        How large of a percentage must the test set be?
  -iv, --include_val    Include a validation set in the generated dataset.
  --inference           Generate a dataset for inference.
  -d, --debug_mode      (reduces the generated dataset size for debugging and testing)
  -c CONFIG, --config CONFIG
                        Specify the path to the config file.

Enjoy the program! :)
```

Example run:
```
python generate_test_train_split.py --languages th --test_set_size .4 --train_set_resolutions 200 --train_set_fontstyles 'normal' -iv
```

### Step 2

Train the model

```
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr"
# Copy test-set over into inference folder and rename
cp -rf train_test_split.csv inference/inference_data.csv
```

The `--model_name` parameter specifies what the name of the model saved on disk should be, e.g. `thaiocr` will save under `./models/thaiocr.pth` along with its one-hot encoder `./models/thaiocr.ohe`.

### Step 3

Perform inference (test_set evaluation)

```
cd inference && python inference.py --batch_size=32 --model_name="thaiocr" && cd -
```

An output should appear reporting precision, recall and F1 for each class, as well as macro- / and weighted average precision/recall/f1 scores for every class in the JSON-format. There is also a CSV table stored of the performance in `.inference`

#### Non-cli Parameters

Additional parameters, e.g. filepaths to the train/test directory and learning_rate can be configured in the config.json or directly in the `modules/TrainingConfig.py` classes respectively.

## Analysis

### Performance Report

| **Training data**                    | **Testing data**                                                                                 | Precision | Recall | F1    | Accuracy |
| ------------------------------------ | ------------------------------------------------------------------------------------------------ | --------- | ------ | ----- | -------- |
| Thai normal text, 200dpi             | Thai normal text, 200dpi                                                                         | 92.54     | 89.83  | 89.46 | 89.74    |
| Thai normal text, 400dpi             | Thai normal text, **200dpi (yes, different resolution, figure out the logistics of this)  <br>** | 93.69     | 92.95  | 93.02 | 93.11    |
| Thai normal text, 400 dpi            | Thai bold text, 400dpi                                                                           | 92.88     | 90.19  | 90.14 | 90.58    |
| Thai bold text                       | Thai normal text                                                                                 | 90.10     | 88.06  | 87.85 | 88.04    |
| All Thai styles                      | All Thai styles                                                                                  | 97.73     | 97.63  | 97.65 | 97.56    |
| Thai and English normal text jointly | Thai and English normal text jointly.                                                            | 96.73     | 95.96  | 95.95 | 95.83    |
| All Thai and English styles jointly. | All Thai and English styles jointly.                                                             | 95.86     | 95.56  | 95.58 | 95.45    |
| English normal, 200dpi               | English bold, 400dpi                                                                             | 87.63     | 83.43  | 82.14 | 83.43    |

All the above numbers are in %.

The **model architecture** used consisted of a convolutional layer followed by ReLU activiation (non-linearity) finalised by a linear layer attempting to predict one-hot encoded character labels.

The most successful model trained (according to the metrics) was the full thai dataset comprising of all thai styles and resolutions. An accuracy and f1 measure of ~97% was achieved. Second best model trained was the jount model focussing only on normal text and then the joint model focussing on all styles. Since the performance difference between the joint-model is quite small, multiple runs would give a better insight under which data the model architecture performs better. All training involved a maximum of 10 epochs and a batch size of 32. All images were automatically resized to a dimension of 128x128 pixels. The CNN-layer kernel-size was 3, stride=1 and the padding=1. No hyperparameter-tuning ocurred. The learning rate was set to .001. Cross Entropy loss was used and SGD was the optimiser. Further, I used early stopping with a threshold of .01 on the loss difference between epochs of the validation set.

In scenarios where the test-set deviates in configuration from the training set I used the entire data. If the test-set and training-set configurations were the same, the test-set contained ~20% of the entire dataset sampled randomly in a stratified way (so each character-class has equal representation in training and test-set).

Looking at the individual experiments For experiment 2, the test set performance on a fraction of the 400dpi dataset itself was 97.45%, whereas an accuracy of 93.11% was achieved over the entire thai-normal-200dpi characters. I would argue this is practically more relevant than training on 400dpi using 200dpi for validation (and early stopping), because we would be interested in seeing how well models generalise trained higher resolution settings to low resolution settings. In experiment 8, I attempted doing the inverse: training in a low resolution setting and checking how much performance can be achieved evaluating on higher resolution (also varying in font style). This lead to the worst performance in all experiments, with roughly 82.14% macro F1. 

### Qualitative Analysis of Experiment 8

Here we provide some analysis of errors for an English configuration.

#### Selected Errors

From the observation that some characters are visually more similar to others for us humans, we want to briefly analyse whether the same applies to the model. This means that the model would be more likely confused (less performance) in the categories capital I vs. small l (L) in English. While looking at the character-wise F1 scores we observe that the lowest performing characters of Experiment 8 are: "l", "V", "W", "S", "I" and "i" (in order of F1 performance; smallest to largest). 

The low F1 scores for the characters above, support the idea that "l" / "I" distinction might be problematic. A look at the confusion matrix however invalidates this hypothesis and suggests that "l"'s true positives get confused "i" and "." often and some "i" and "j" end up being classified as "l".  "I" ends up being classified as ".", "i" and much less than expected as "l" (only 7 times). Most false positives in the "I" category are from T(46) and Y(12). Therefore our initial hypothesis that I/l are mixed up often cannot be supported and there is no indication of a "symmetry" in misclassification between the two.

While precision is perfect (F1: 1.00) for S and V, their performance in F1 is impeded by a low recall. This holds, if also to a lesser extent, for "W" and "X". The inverse can be observed for some lowercase counterparts "w" and "x", as well as additionally, "t" and "y" have relatively low precision values but a high recall. Looking at the confusion matrix we see that S, V as well as W and X appear to loose their "recall" to the lowercase characters. `S -> t`, `V -> w`, `W -> x`, `X -> y`. My hypothesis is that this could somehow be related to the convolutional operator that cannot handle the distinctions between these letters well. I would not attribute it to the additional boldness of the characters or the resolution, for the effect is also observable on the test set of non-varied test-set of `en-normal-200dpi`, although to a lesser extent.

A skim through Thai alphabet characters reveals that there might be similar performance challenges differentiating between the characters ข (Kho Khai) vs. ฃ (Kho Khat) than other characters.

### Replication Instructions

#### Experiment 1: `thai-normal-200dpi -> thai-normal-200dpi`
```
python generate_test_train_split.py --languages th --test_set_size .4 --train_set_resolutions 200 --train_set_fontstyles 'normal' -iv
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment1"
cp -rf train_test_split.csv inference/inference_data.csv
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment1" && cd -
```

#### Experiment 2: `thai-normal-400dpi -> thai-normal-200dpi`
```
python generate_test_train_split.py --languages th --test_set_size .4 --train_set_resolutions 400 --train_set_fontstyles 'normal' -iv
python generate_test_train_split.py --languages th --train_set_resolutions 200 --train_set_fontstyles 'normal' --inference
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment2"
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment2" && cd -
```

#### Experiment 3: `thai-normal-400dpi` -> thai-bold-400dpi`

```
python generate_test_train_split.py --languages th --test_set_size .4 --train_set_resolutions 400 --train_set_fontstyles 'normal' -iv
python generate_test_train_split.py --languages th --train_set_resolutions 400 --train_set_fontstyles 'bold' --inference
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment3"
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment3" && cd -
```

#### Experiment 4: `thai-bold-*dpi -> thai-normal-*dpi`
```
python generate_test_train_split.py --languages th --test_set_size .4 --train_set_resolutions 200 300 400 --train_set_fontstyles 'bold' -iv
python generate_test_train_split.py --languages th --train_set_resolutions 200 300 400 --train_set_fontstyles 'normal' --inference
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment4"
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment4" && cd -
```

#### Experiment 5: `thai-*-* -> thai-*-*`
```
python generate_test_train_split.py --languages th --test_set_size .4 --train_set_resolutions 200 300 400 --train_set_fontstyles 'bold' 'bold_italic' 'italic' 'normal' -iv
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment5"
cp -rf train_test_split.csv inference/inference_data.csv
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment5" && cd -
```

#### Experiment 6: `thai/english-normal-*dpi -> thai/english-normal-*dpi`
```
python generate_test_train_split.py --languages en th --test_set_size .4 --train_set_resolutions 200 300 400 --train_set_fontstyles 'normal' -iv
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment6"
cp -rf train_test_split.csv inference/inference_data.csv
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment6" && cd -
```

#### Experiment 7: `thai/english-*-* -> thai/english-*-*`
```
python generate_test_train_split.py --languages en th --test_set_size .4 --train_set_resolutions 200 300 400 --train_set_fontstyles 'bold' 'bold_italic' 'italic' 'normal' -iv
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment7"
cp -rf train_test_split.csv inference/inference_data.csv
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment7" && cd -
```

#### Experiment 8: `english-normal-200dpi -> english-bold-400dpi`
```
python generate_test_train_split.py --languages en --test_set_size .4 --train_set_resolutions 200 --train_set_fontstyles 'normal' -iv
python generate_test_train_split.py --languages en --train_set_resolutions 400 --train_set_fontstyles 'bold' --inference
python train.py --batch_size=32 --epochs=10 --model_name="thaiocr_experiment8"
cd inference && python inference.py --batch_size=32 --model_name="thaiocr_experiment8" && cd -
```

## Bonus
Not completed so far.