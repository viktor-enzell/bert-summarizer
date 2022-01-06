# Dataless Text Classification on AG News with BERT
Explotiation of dataless text classification with a small BERT model on the AG News topic dataset. News articles and category labels are embedded using BERT. Similarity between article embeddings and label embeddings is used as a baseline approach and several experiments to improve accuracy are conducted. Additionally, the same BERT model is fine-tuned on the full dataset for supervised comparison. With dataless classification an accuracy of 77.6% is achieved whereas the fine-tuned BERT model achieves an accuracy of 91%.

### Running the experiments
Install the required packages with `pip install -r requirements.txt` and run the notebooks in `src/fine_tuning` and `src/dataless` respectively.
