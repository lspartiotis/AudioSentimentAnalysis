# AudioSentimentAnalysis
defining sentiments from audio using machine learning


# Project_Name: Audio Sentiment Classification

Leandros Atherinis-Spartiotis
Vicky Polymeropoulou
Chaidw Poulianou

## Description

We detect the sentiments that songs of a wide variety of genres bring up.
<ol>
  <li>happy</li>
  <li>calm</li>
  <li>anger</li>
  <li>sad</li>
</ol>

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Provide step-by-step instructions on how to set up the project environment. Mention any prerequisites, dependencies, and how to install them.

```bash
git clone https://github.com/yourusername/Project_Name.git
cd Project_Name
pip install -r requirements.txt
```
## Run

Various global parameters setting the configurations in the program are located in the file `params.py`.

Our final and best model is a RandomForset with KFold cross validation and GridSearch which uses SMOTE for training samples oversampling.
To run this model, you can execute the `pipeline_os_rf.py` file.
