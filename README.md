# F.CSC336NLP
Author: Э.Мичидмаа
Course: Natural Language Processing
Date: 2025-12-19

# 1. Оршил
Энэхүү судалгааны ажлын хүрээнд sentiment analysis асуудлыг сонгон авч, түгээмэл ашиглагддаг нэгэн dataset дээр уламжлалт машин сургалт болон гүн сургалтын аргуудын гүйцэтгэлийг харьцуулан судаллаа. Судалгааны үндсэн зорилго нь текст өгөгдлөөс embedding үүсгэж, тэдгээрийг ашиглан ангиллын (classification) загварууд сургах, мөн embedding болон модель сонголт нь үр дүнд хэрхэн нөлөөлж байгааг системтэйгээр үнэлэхэд оршино. Үүний тулд dataset-ийн онцлогийг нарийвчлан судалж, өмнөх судалгаануудын арга барилыг нэгтгэн дүгнэсэн бөгөөд үр дүнг стандарт үнэлгээний хэмжүүрүүдээр харьцуулан шинжилсэн. Төслийн хүрээнд TF-IDF, Word2Vec, BERT зэрэг embedding аргуудыг ашиглаж, Logistic Regression, Random Forest, AdaBoost зэрэг уламжлалт загваруудыг LSTM зэрэг гүн сургалтын загвартай харьцуулан туршиж, тэдгээрийн давуу болон сул талыг бодит туршилтын үр дүнд тулгуурлан үнэлэв.

# 2. Бодлогын даалгавар (Tasks)
1. Текст өгөгдлийг crawl хийж цуглуулах
2. Өгөгдлийг цэвэрлэх (preprocessing)
3. Embedding үүсгэх (TF-IDF, Word2Vec, BERT)
4. Загвар сургах (Machine Learning & Deep Learning)
5. Embedding-үүдийн хоорондын харьцуулалт хийх
6. Үр дүнг үнэлэх (Accuracy, Precision, Recall, F1-score)

# 3. Dataset-ийн танилцуулга (Dataset Description)
- Dataset нэр: Stanford Large Movie Review Dataset (IMDB movie reviews)
- Dataset зорилго: Binary (хоёр ангилал) sentiment буюу сэтгэл хөдлөлийн ангилал хийх (текстээс тухайн сэтгэгдэл эерэг эсвэл сөрөг эсэхийг тодорхойлох)
- Өгөгдлийн төрөл: Text classification
- Ангиллын тоо: 2 (Positive, Negative)
- Нийт өгөгдлийн хэмжээ: 50,000 мөр (25,000 сургалтын датa болон 25,000 тест датa). Нэмэлт unlabeled (шошгогүй) 50,000 баримт (unsupervised learning-д ашиглах боломжтой).
## 3.1 Data эх сурвалж
- Dataset link: [https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz]
- Dataset paper: [https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf]

## 3.2 Dataset-тэй холбоотой судалгааны өгүүлэл
**Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews – Grégoire Mesnil, Tomas Mikolov, Marc’Aurelio Ranzato, Yoshua Bengio, 2014. [https://arxiv.org/abs/1412.5335?utm_source=chatgpt.com]**
 Энэхүү судалгаанд кино шүүмжийн сантимент ангилалд generative болон discriminative аргуудыг хослуулан ашиглах нь гүйцэтгэлийг сайжруулж чадах эсэхийг судалсан.
 Ашигласан аргууд: 
   - Word2Vec (neural word embeddings)
   - Logistic Regression
   - Neural Network classifier
   - Ensemble learning
Үр дүн:
  - Ensemble загвар нь дан classifier-аас илүү accuracy үзүүлсэн
  - Word embedding ашигласан загварууд traditional bag-of-words-оос илүү сайн байсан
Дүгнэлт:
  Generative болон discriminative аргуудыг хослуулах нь текстийн семантик мэдээллийг илүү сайн барьж, сантимент ангиллын чанарыг сайжруулдаг.
**Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis – Gibson Nkhata et al., 2025. [https://arxiv.org/abs/2502.20682?utm_source=chatgpt.com]**
BERT-ийн contextual embedding-ийг LSTM-тай хослуулж fine-grained sentiment classification хийх зорилготой.
Ашигласан аргууд: 
  - BERT embedding
  - Bidirectional LSTM
  - Fine-tuning
Үр дүн:
  - BERT + BiLSTM загвар нь зөвхөн BERT-ээс илүү F1-score үзүүлсэн
  - Context-aware embedding илүү нарийн sentiment ялгадаг
Дүгнэлт:
  Transformer болон RNN-г хослуулах нь урт өгүүлбэр дэх sentiment илрүүлэхэд илүү үр дүнтэй.
**Enhancing Sentiment Classification with Machine Learning and Combinatorial Fusion – Sean Patten et al., 2025. [https://arxiv.org/abs/2510.27014?utm_source=chatgpt.com]**
Олон машин сургалтын аргуудын үр дүнг combinatorial fusion ашиглан нэгтгэх замаар сантимент ангиллыг сайжруулахыг зорьсон.
Ашигласан аргууд: 
  - Logistic Regression
  - Random Forest
  - SVM
  - Ensemble / Fusion techniques
Үр дүн:
  - Fusion ашигласан загварууд accuracy-гаар дан загваруудаас давсан
  - Model diversity үр дүнд эерэг нөлөө үзүүлсэн
Дүгнэлт:
  Classifier-уудын ялгаатай алдааг нөхөх замаар ensemble арга илүү тогтвортой үр дүн өгдөг.
**Semantic Sentiment Analysis Based on Probabilistic Graphical Models and RNN – Ukachi Osisiogu, 2020. [https://arxiv.org/abs/2009.00234?utm_source=chatgpt.com]**
Семантик хамаарлыг илүү сайн ойлгохын тулд probabilistic graphical model болон RNN-г хослуулсан.
Ашигласан аргууд: 
  - Recurrent Neural Network
  - Probabilistic Graphical Models
Үр дүн:
  - Semantic-level sentiment илүү сайн ялгасан
  - Traditional ML-ээс илүү F1-score үзүүлсэн
Дүгнэлт:
  Семантик хамаарлыг explicit загварчлах нь сантимент анализын чанарыг сайжруулдаг.
**A Comprehensive Benchmarking Pipeline for Transformer-Based Sentiment Analysis using Cross-Validated Metrics – Dodo Zaenal Abidin et al., 2025. [https://www.researchgate.net/publication/381285499_Sentiment_Analysis_of_IMDb_Movie_Reviews?utm_source=chatgpt.com]**
Transformer-based загваруудын гүйцэтгэлийг стандарт pipeline ашиглан харьцуулсан.
Ашигласан аргууд: 
  - BERT
  - RoBERTa
  - DistilBERT
Үр дүн:
  - RoBERTa BERT-ээс илүү дүн үзүүлсэн
  - Cross-validation нь үнэлгээг илүү найдвартай болгосон
Дүгнэлт:
  Transformer загваруудын гүйцэтгэл нь pretraining strategy-оос ихээхэн хамаардаг.
**Sentiment Analysis of IMDB Movie Reviews Using BERT (IARIA conference-ийн paper, 2023). [https://personales.upv.es/thinkmind/dl/conferences/eknow/eknow_2023/eknow_2023_2_30_60011.pdf?utm_source=chatgpt.com]**
IMDb датаг ашиглан BERT-ийн сантимент ангиллын чадварыг үнэлсэн
Ашигласан аргууд: 
  - Softmax classifier
  - BERT
Үр дүн:
  - *Accuracy ~90%+
  - Traditional ML-ээс илүү үр дүнтэй
Дүгнэлт:
  Contextual embedding нь sentiment classification-д маш чухал.
**Sentiment Analysis of IMDb Movie Reviews (thesis / Kaggle based) – Domadula 2023. [https://www.diva-portal.org/smash/get/diva2%3A1779708/FULLTEXT02.pdf?utm_source=chatgpt.com]**
Kaggle болон IMDb dataset дээр олон аргыг харьцуулсан судалгаа.
Ашигласан аргууд: 
  - Logistic Regression
  - LSTM
  - Word2Vec
Үр дүн:
  - LSTM + Word2Vec хамгийн сайн үр дүн үзүүлсэн
Дүгнэлт:
  Sequence model нь текстийн дарааллыг илүү сайн ойлгодог.
**Sentiment Analysis of IMDb Movie Reviews Using LSTM (conference/study) – Saad Tayef (document). [https://www.scribd.com/document/744893212/Sentiment-Analysis-of-IMDb-Movie-Reviews-Using-LSTM?utm_source=chatgpt.com]**
LSTM ашиглан урт текстийн sentiment тодорхойлох боломжийг судалсан.
Ашигласан аргууд: 
  - Word2Vec
  - LSTM
Үр дүн:
  - Accuracy ~88–90%
Дүгнэлт:
  LSTM нь урт өгүүлбэр дэх контекстийг хадгалах чадвартай.

Өмнөх судалгаануудыг шинжилж үзэхэд sentiment analysis асуудлыг шийдвэрлэхэд уламжлалт машин сургалт болон гүн сургалтын аргуудыг өргөнөөр ашигласан нь ажиглагддаг. Тухайлбал, текстийг TF-IDF embedding-ээр төлөөлүүлж, Logistic Regression загвартай хослуулсан арга нь суурь шийдэл (baseline) болгон түгээмэл хэрэглэгддэг. Үүнээс гадна Word2Vec embedding-д суурилсан CNN болон LSTM загваруудыг ашиглан дараалал болон орон зайн мэдээллийг хамтад нь авч үзэх хандлага өргөн тархсан байна. Мөн GloVe, Word2Vec зэрэг урьдчилан сургагдсан embedding-үүдийг ашигласнаар сургалтын өгөгдөл харьцангуй бага нөхцөлд ч загварын гүйцэтгэлийг сайжруулах боломжтойг олон судалгаанд онцолсон байдаг. Сүүлийн жилүүдэд Transformer архитектурт суурилсан BERT, GPT зэрэг загварууд текстийн контекстийг илүү гүнзгий ойлгох чадвараараа sentiment analysis даалгаварт өндөр үр дүн үзүүлж, орчин үеийн судалгааны гол чиг хандлага болоод байна.

# 4. Судалгаанд ашигласан арга зүй 
Энэхүү судалгаанд sentiment analysis хийх зорилгоор олон төрлийн word embedding болон машин сургалтын загварууд-ыг системтэйгээр харьцуулан туршсан. Судалгааны үндсэн зорилго нь embedding-ийн төрөл болон загварын сонголт нь sentiment analysis-ийн гүйцэтгэлд хэрхэн нөлөөлж байгааг тодорхойлох явдал байв.

## 4.1 Ашигласан embedding аргууд
Судалгаанд дараах 7 төрлийн word embedding-үүдийг туршсан:

### 4.1.1 Transformer-д суурилсан contextual embeddings
BERT (Bidirectional Encoder Representations from Transformers)

BERT-Base (uncased): Том жижиг үсгийг ялгахгүй, 768 dimension

BERT-Base (cased): Том жижиг үсгийг ялгадаг, 768 dimension

RoBERTa (Robustly Optimized BERT): BERT-ийн сайжруулсан хувилбар, 768 dimension

ALBERT (A Lite BERT): Parameter багатай, үр ашигтай BERT хувилбар, 768 dimension

SBERT (Sentence-BERT): Өгүүлбэрийн түвшний embedding, 384 dimension

HateBERT: Hate speech болон сөрөг контекст дээр тусгайлан сургасан BERT, 768 dimension

### 4.1.2 Уламжлалт word embedding
Word2Vec: CBOW болон Skip-gram архитектур ашиглан сургасан, 100 dimension
Бүх Transformer-д суурилсан embedding-үүд нь pre-trained байсан бөгөөд тухайн загваруудыг feature extractor болгон ашиглаж, CLS token-ийн эцсийн давхаргын төлөөллийг баримт бичгийн embedding болгон авсан. Word2Vec embedding-ийн хувьд IMDB dataset-ийн өгөгдөл дээр шууд сургаж, баримт бичиг бүрийн үгсийн embeddings-үүдийн дунджийг авах замаар баримт бичгийн төлөөллийг үүсгэсэн.

## 4.2 Ашигласан машин сургалтын загварууд
Embedding бүрийн гүйцэтгэлийг үнэлэхийн тулд дараах 4 төрлийн машин сургалтын загвар-ыг ашигласан:

### 4.2.1 Logistic Regression
Logistic Regression нь текст ангиллын суурь (baseline) загвар болгон өргөн ашиглагддаг. Энэхүү судалгаанд GridSearchCV ашиглан hyperparameter-үүдийг оновчтой болгосон:

C (regularization strength): [0.01, 0.1, 1, 10]
solver: ['liblinear', 'saga']
max_iter: [500, 1000]
Загварын магадлал:

[ P(y=1 \mid x) = \frac{1}{1 + e^{-w^T x}} ]

### 4.2.2 Random Forest
Random Forest нь олон тооны decision tree-үүд дээр суурилсан ensemble загвар юм. Дараах hyperparameter-үүдийг оновчтой болгосон:

n_estimators: [50, 100, 200, 300, 400]
max_depth: [10, 20]
min_samples_split: [2, 5]
min_samples_leaf: [1]
max_features: ['sqrt']
### 4.2.3 AdaBoost
AdaBoost нь boosting аргачлалд суурилсан ensemble загвар бөгөөд DecisionTreeClassifier-ийг үндсэн ангилагч (base estimator) болгон ашигласан:

n_estimators: [50, 100, 200, 300]
learning_rate: [0.01, 0.1, 0.5, 1.0]
algorithm: ['SAMME']
### 4.2.4 LSTM (Long Short-Term Memory)
LSTM нь дарааллын өгөгдлийн урт хугацааны хамаарлыг сурахад илүү үр ашигтай гүн сургалтын загвар юм. TensorFlow/Keras ашиглан дараах архитектуруудыг туршсан:

LSTM units: [128, 256]
Dropout rate: [0.3, 0.5]
Bidirectional LSTM: [True, False]
Optimizer: Adam (learning_rate=0.001)
Loss function: Binary crossentropy
LSTM нь 3D тогтолцоон (samples, timesteps, features) шаарддаг тул Word2Vec embedding-ийг өгүүлбэр түвшинд дарааллын хэлбэрт шилжүүлж ашигласан.

## 4.3 Өгөгдөл боловсруулалт ба сургалтын pipeline
Судалгааны нийт процесс дараах дарааллаар явагдсан:

### 4.3.1 Өгөгдөл цэвэрлэх (Data Cleaning)
HTML тэмдэгтүүд болон тусгай тэмдэгтүүдийг арилгах
Текстийг жижиг үсэгт шилжүүлэх (lowercase)
Stopwords устгах (Word2Vec-ийн хувьд)
Tokenization хийх
### 4.3.2 Embedding үүсгэх
Transformer embeddings: Pre-trained загварууд ашиглан CLS token-ийн төлөөллийг гаргаж авах
Word2Vec: IMDB dataset дээр сургаж, баримт бичгийн үгсийн embedding-үүдийн дунджийг авах
### 4.3.3 Өгөгдлийг хуваах
Сургалтын өгөгдөл: 40,000 samples (80%)
Тестийн өгөгдөл: 10,000 samples (20%)
Ангилал тэнцвэртэй (эерэг 50%, сөрөг 50%)
### 4.3.4 Загвар сургах ба үнэлэх
GridSearchCV ашиглан 5-fold cross-validation хийх
Хамгийн сайн hyperparameter-үүдийг сонгох
Тестийн өгөгдөл дээр эцсийн гүйцэтгэлийг үнэлэх
Classification report, confusion matrix үүсгэх
## 4.4 Үнэлгээний хэмжүүрүүд
Загваруудын гүйцэтгэлийг дараах хэмжүүрүүдээр үнэлсэн:

Accuracy: Нийт зөв таамаглагдсан ангиллын харьцаа
Precision: Эерэг гэж таамагласан зүйлсийн яг хэдэн нь үнэхээр эерэг байсан
Recall: Бүх эерэг зүйлсийн хэдэн хувийг зөв олж таньсан
F1-score: Precision болон Recall-ын гармоник дундаж
Cross-validation score: Сургалтын өгөгдөл дээрх дунджийн гүйцэтгэл
Training time: Загвар сургахад зарцуулсан хугацаа (секунд)
## 4.5 Судалгааны туршилтын дизайн
Судалгааг дараах байдлаар зохион байгуулсан:

Эмбеддингийн харьцуулалт: 7 төрлийн embedding бүрийг 4 загвартай туршиж, нийт 28 туршилт хийх
Hyperparameter оновчлол: GridSearchCV ашиглан загвар бүрийн хамгийн сайн параметрүүдийг автоматаар олох
Тогтвортой үр дүн: Random seed (42) ашиглан үр дүнг давтагдах боломжтой болгох
Системтэй бүртгэл: Туршилт бүрийн үр дүнг CSV файл болон log файлд нарийвчлан хадгалах
## 4.6 Ашигласан embedding-үүдийн техник мэдээлэл
| Embedding        | Dimension | Төрөл       | Онцлог |
|------------------|-----------|-------------|--------|
| BERT (uncased)   | 768       | Contextual  | Том, жижиг үсэг ялгахгүй, bidirectional |
| BERT (cased)     | 768       | Contextual  | Том, жижиг үсэг ялгадаг, bidirectional |
| RoBERTa          | 768       | Contextual  | BERT-ээс илүү их өгөгдөл, сайжруулсан сургалт |
| ALBERT           | 768       | Contextual  | Parameter багатай, parameter sharing |
| SBERT            | 384       | Contextual  | Өгүүлбэрийн ижилтэт байдлын даалгаварт тохирсон |
| HateBERT         | 768       | Contextual  | Hate speech, сөрөг текст дээр тусгайлан сургасан |
| Word2Vec         | 100       | Static      | IMDB dataset дээр сургасан, bag-of-words |

# 5.1 Сургалтын орчин болон төхөөрөмж
Google Colab орчин:

CPU: Intel Xeon (vCPU, Colab-ээс олгогддог)

RAM: ~12.7 GB стандарт / ~25.5 GB high-RAM боломжтой (runtime тохиргоогоор сонгож болно)

GPU: NVIDIA Tesla T4 

CUDA Version: 12.1 (тухайн runtime-ээс хамаарна)

CUDA Cores: ~2,560 (T4-ийн хувьд)

Memory: 16 GB GDDR6 (V100) / 15 GB (T4)

OS: Ubuntu 20.04 (Linux)

Python: 3.10+ 

Frameworks:

 - scikit-learn

 - numpy

 - pandas

 - PyTorch 2.0+ (CUDA-тэй)

 - TensorFlow 2.x

 - XGBoost (GPU-enabled хувилбар Colab дээр суулгах боломжтой)

 - Transformers (HuggingFace)

Онцлог:

 - Cloud дээр ажиллаж байгаа тул hardware-н хүчин чадал нь тухайн Colab runtime-с хамаарна.
 - Хувийн төхөөрөмж шаардлагагүй, интернэттэй байх л хангалттай.
