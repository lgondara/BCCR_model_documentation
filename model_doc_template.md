---
model-index:
- name: Reportability filter
license: mit
datasets:
- A random sample of 40,000 reports marked reportable by eMarC
language:
- en
base_model: emilyalsentzer/Bio_ClinicalBERT
---


<img src="http://www.phsa.ca/_layouts/15/CUSTOM/EWI/assets/img/phsa/logo.png" alt="PHSA Logo" style="margin-left:'auto' margin-right:'auto' display:'block'"/>


# Model summary

Reportability filter model is traiend to distinguish between true reportable tumor pathology reports and the false positive tumor pathology reports. 
The input data to the model are the pathology reports marked reportable by eMaRC, out of which 20-30% are false positives.

## Model description

- **Model type:** Finetuned version of BioClinical BERT, which is initially a finetuned version of BioBERT, finetuned using all notes from MIMIC III,
  a database containing electronic health records from ICU patients at the Beth Israel Hospital in Boston, MA.
- **Language(s) (NLP):** Primarily English
- **License:** MIT
- **Finetuned from model:** [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

### Model Sources

- **Local directory:** svmonc02//...
- **GitHub repository:** None
- **Huggingface repository:** None


### Training data details
Training data is a random sample from the diagnosis year 2021, and includes 40,000 pathology reports initially marked reportable by the rule-based eMaRC system. These reports are then manually reviewed by tumor registrars (TRs) to assign a label of true positive (a report containing reportable tumor pathology) or false positive (a report missing reportable tumor pathology). Being a large random sample, we assume it to be representative of the population.

A random training/test split of 25,000/15,000 is used for training and testing the model using the following code.

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(train_data, test_size=0.375,random_state=100) 
```

### Data preprocessing
All reports are preprocessed by removing special characters ("X0D", ".br") and then are passed through a longformer model [valhalla/longformer-base-4096-finetuned-squadv1](https://huggingface.co/valhalla/longformer-base-4096-finetuned-squadv1) to extract relevant sections using the following question-answering approach.

- "Comment": "Can you extract the section which immediately follows a subheading containing 'comment' and exclude sections that occur after the next subheading?"
- "Addendum": "Can you extract the section which immediately follows a subheading containing 'addendum' and exclude sections that occur after the next subheading?"
- "Gross description": "Can you extract the section which immediately follows a subheading containing 'gross description' and exclude sections that occur after the next subheading?"
- "Diagnosis": "Can you extract only the section that describes the diagnosis and not a section that appears before or after the diagnosis?"
- "Clinical History": "Can you extract the section which immediately follows subheadings related to 'clinical history' or 'clinical information' and exclude sections that occur after the next subheading"
- "Microscopic": "Can you extract the section which immediately follows a subheading related to 'microscopic' and exclude sections that occur after the next subheading?"
- "Overall report": "Can you extract the overall report without metadata?"

The reportability filter model is then trained on the truncated version of relevant report segment, truncated to ensure that the input complies with 512 token limitations of the base model (BERT). 


### Model input
Text after the preprocessing step. Example

"Patient Information:
The patient, [Patient's Name], [Age], presented with [Clinical History]. The medical record number is [MRN], and the procedure was conducted on [Date].

Specimen Received:
Tissue from a tumor located in [Anatomical Site] was received for examination, identified by Specimen ID [Specimen ID Number].

Gross Description:
The specimen consists of a [size], [shape], [color] tumor measuring approximately [dimensions] in the [anatomical location]. The tumor appears to be [description of tumor boundaries] and shows signs of [any visible invasion or relationship to surrounding tissues].

Microscopic Examination:
Histologically, the tumor presents as [Histological Type], graded as [Grade, if applicable]. Margins are [Description of margins - clear, involved, close, etc.]. The tumor measures [Size] and exhibits [presence/absence] of lymphovascular invasion. The mitotic rate is [Number of mitotic figures per high power field]. Immunohistochemistry showed [Results of specific stains performed].

Diagnosis:
The primary diagnosis is [Primary Diagnosis], with additional findings of [Additional Findings or Features] observed during examination.

Pathologist's Comments:
Further examination revealed [Pathologist's observations, any recommendations, or pertinent remarks based on the findings].

Final Diagnosis:
In summary, the diagnosis is [Summarized Diagnosis], considering all the histopathological features and additional observations, indicating [specific staging or grading information if applicable]."


## Model output
Prediction if either the report is reportable or non-reportable (1 or 0) along with the model confidence score (probability from 0-1).

## Intended uses & limitations

The model is finetuned on a random sample of 25000 pathology reports from the diagnosis year 2021. The model should only be used to distinguish between reportable and non-reportable tumor pathology reports. The model's performance can vary depending on the input data format, preprocessing, data distribution shift (concept drift), etc. The performance metrics provided below are on the test set only and should not be used for decision-making regarding the model deployment in a general scenario.

The model can be run using the following syntax for inference.

```python
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import os

model_dir = "model_directory"
model_name = 'fp_model'
num_labels = 2

# Input data preprocessed and ready for inference
df = input_data

model = AutoModelForSequenceClassification.from_pretrained(model_dir + '\\' + model_name , num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_len = 512)

def class_model_inference(text, section_type, class_pipe = class_pipe):
    if section_type == 'entire report':
        return(['reportable', 1.0])

    query = class_pipe(text, truncation = True)
    pred_label = label_lookup[query[0]['label']]
    return ([pred_label, query[0]['score']])

df[['predicted_label', 'model_score']] = df.apply(lambda x: pd.Series(class_model_inference(text = x['final_text'], section_type=x['section_type'])), axis = 1)
```

## Bias, Risks, and Limitations

Where the longformer segmentation model cannot find a specific segment and we use the entire report as the model input, we assign that report to the reportable class. This is done because during intitial investigation we found that the reports with missing xx section were almost all reportable.

The impact of this approach is that if we use an input data of different format where the longformer model fails and the model ends up using the entire report, we might have a large number of false positives.


Thresholding details if any:


## Training and evaluation data

The reportability filter achieves the following performance on test data.

- Overall accuracy: 0.xx
- True reportable accuracy: 0.xx
- False positive accuracy: 0.xx


### Training hyperparameters

The following hyperparameters were used during training:

- learning_rate: 2e-05
- per_device_train_batch_size: 16
- per_device_eval_batch_size: 16
- seed: xx
- distributed_type: none, single GPU
- num_train_epochs: 2
- weight_decay: 0.01


### Framework versions

- Transformers 4.34.0
- Pytorch 2.0.1+cu118
- xxx
