# IndoNLP-2024
<h1>From Scarcity to Capability: Empowering Fake News Detection in Low-Resource Languages with LLMs</h1>
<p> The rapid spread of fake news presents a significant global challenge, particularly in low-resource languages like Bangla, which lack adequate datasets and detection tools. Although manual fact-checking is accurate, it is expensive and slow to prevent the dissemination of fake news. Addressing this gap, we introduce BanFakeNews-2.0, a robust dataset to enhance Bangla fake news detection. This version includes 11,700 additional, meticulously curated fake news articles validated from credible sources, creating a proportional dataset of 47,000 authentic and 13,000 fake news items across 13 categories. In addition, we created a manually curated independent test set of 460 fake and 540 authentic news items for rigorous evaluation. We invest efforts in collecting fake news from credible sources and manually verified while preserving the linguistic richness. We develop a benchmark system utilizing transformer-based architectures, including fine-tuned Bidirectional Encoder Representations from Transformers variants (F1-87%) and Large Language Models with Quantized Low-Rank Approximation (F1-89%), that significantly outperforms traditional methods. BanFakeNews-2.0 offers a valuable resource to advance research and application in fake news detection for low-resourced languages. </p>

<h3>The following link is directed to our BanFakeNews-2.0 dataset which is uploaded in Kaggle platform. We have annotated our authentic news as 1 and fake news as 0</h3>
<href>https://www.kaggle.com/datasets/hrithikmajumdar/bangla-fake-news</href>
<h3>The doi link for the BanFakeNews-2.0 dataset is given below which we have published in the Mendeley which is a dataset sharing platform.</h3>
<href>https://data.mendeley.com/datasets/kjh887ct4j/1</href>

<h3>The main dataset, BanFakeNews-2.0 is splitted into train, test and validation sets which is under the dataset folder. Moreover, the new test dataset for rigorous evaluation consisting 460 Fake news and 540 Real news articles is named as New_Test_Dataset.csv in the main folder.</h3>

<h3>Traditional Linguistic Features with SVM:</h3>
<p>In the FakeNews-master folder , we have actually experimented our classical machine learning model(SVM) preprocessed with linguistic features named as Unigram, Bigram, Trigram and C3, C4 and C5 gram.</p>
<h4>Basic Experiments</h4>
<ul type=disk>
<li>Go to FakeNews-master/Models/Basic folder</li>
<li>Use <b>python n-gram.py [Experiment Name] [Model] [-s](optional)</b> to run an experiment. For example: `python n-gram.py Emb_F SVM -s` will run the Emb_F experiment using SVM Model. Use -s to Save the results. 
<h4>Experiment Names</h4>(Please follow the paper to read the details about experiments) : 
   <ul type=star> 
    <li>Unigram</li>
    <li>Bigram</li>
    <li>Trigram</li>
    <li>U+B+T</li>
    <li>C3-gram</li>
    <li>C4-gram</li>
    <li>C5-gram</li>
    <li>C3+C4+C5</li>
    <li>Embedding</li>
    <li>all_features</li>
   </ul>
<h4>Models</h4>
    <ul type=star>
       <li>SVM (Support Vector Machine)</li>
    </ul>
</ul>

<h3>BERT model training notebooks of Table: 3</h3>
<p>These notebooks have the following naming convention: "training with FakeNews <model_name>.ipynb"</p>

<h3>Training SVM with Embedding features and All Features notebook of Table: 3</h3>
<p>Embedding feature notebook name: "Fasttext_svm.ipynb"</p>
<p>All features notebook name: "all-features-svm-c-1-degree-3.ipynb"</p>

<h3>BLOOM and Phi-3 mini training notebook of Table: 3</h3>
<p>BLOOM notebook name: "bloom-banfakenews1 (1).ipynb"</p>

<p>Phi-3 mini notebook name: "phi3-mini-banfakenews2.xpynb"</p>

<h3>Table: 4 notebook name descriptions</h3>
<o>"Banfakenews1" represents BanFakeNews Dataset</p>
<o>"Banfakenews2" represents BanFakeNews-2.0 Dataset</p>
<o>"Newnews" represents tested with external Dataset</p>

## Citation

If you use this code or BanFakeNews-2.0 dataset in your work, please cite the following paper:
```
@inproceedings{shibu-etal-2025-scarcity,
    title = "From Scarcity to Capability: Empowering Fake News Detection in Low-Resource Languages with {LLM}s",
    author = "Shibu, Hrithik Majumdar  and
      Datta, Shrestha  and
      Miah, Md. Sumon  and
      Sami, Nasrullah  and
      Chowdhury, Mahruba Sharmin  and
      Islam, Md Saiful",
    editor = "Weerasinghe, Ruvan  and
      Anuradha, Isuri  and
      Sumanathilaka, Deshan",
    booktitle = "Proceedings of the First Workshop on Natural Language Processing for Indo-Aryan and Dravidian Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.indonlp-1.12/",
    pages = "100--107",
    abstract = "The rapid spread of fake news presents a significant global challenge, particularly in low-resource languages like Bangla, which lack adequate datasets and detection tools. Although manual fact-checking is accurate, it is expensive and slow to prevent the dissemination of fake news. Addressing this gap, we introduce BanFakeNews-2.0, a robust dataset to enhance Bangla fake news detection. This version includes 11,700 additional, meticulously curated fake news articles validated from credible sources, creating a proportional dataset of 47,000 authentic and 13,000 fake news items across 13 categories. In addition, we created a manually curated independent test set of 460 fake and 540 authentic news items for rigorous evaluation. We invest efforts in collecting fake news from credible sources and manually verified while preserving the linguistic richness. We develop a benchmark system utilizing transformer-based architectures, including fine-tuned Bidirectional Encoder Representations from Transformers variants (F1-87{\%}) and Large Language Models with Quantized Low-Rank Approximation (F1-89{\%}), that significantly outperforms traditional methods. BanFakeNews-2.0 offers a valuable resource to advance research and application in fake news detection for low-resourced languages. We publicly release our dataset and model on GitHub to foster research in this direction."
}
```
