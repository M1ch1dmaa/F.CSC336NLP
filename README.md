# F.CSC336NLP
Author: Э.Мичидмаа
Course: Natural Language Processing
Date: 2025-12-19

1. Оршил (Introduction)
Энэ төслийн зорилго нь текст өгөгдөл дээр embedding үүсгэж, тэдгээрийг ашиглан ангиллын (classification) загварууд сургах, мөн classical machine learning болон deep learning аргуудын гүйцэтгэлийг харьцуулах явдал юм.
Төслийн хүрээнд TF-IDF, Word2Vec, BERT embedding аргуудыг ашиглаж, Logistic Regression, Random Forest, AdaBoost, LSTM зэрэг загваруудыг туршив.

2. Бодлогын даалгавар (Tasks)
Текст өгөгдлийг crawl хийж цуглуулах
Өгөгдлийг цэвэрлэх (preprocessing)
Embedding үүсгэх (TF-IDF, Word2Vec, BERT)
Загвар сургах (Machine Learning & Deep Learning)
Embedding-үүдийн хоорондын харьцуулалт хийх
Үр дүнг үнэлэх (Accuracy, Precision, Recall, F1-score)

3. Dataset-ийн танилцуулга (Dataset Description)
Dataset нэр: Stanford Large Movie Review Dataset (IMDB movie reviews)
Dataset зорилго: Binary (хоёр ангилал) sentiment буюу сэтгэл хөдлөлийн ангилал хийх (текстээс тухайн сэтгэгдэл эерэг эсвэл сөрөг эсэхийг тодорхойлох)
Өгөгдлийн төрөл: Text classification
Нийт өгөгдлийн хэмжээ: 50,000 мөр (25,000 сургалтын датa болон 25,000 тест датa). Нэмэлт unlabeled (шошгогүй) 50,000 баримт (unsupervised learning-д ашиглах боломжтой).

Dataset link: [https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz]
Dataset paper: [https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf]

Dataset-тэй холбоотой судалгааны өгүүлэл
1 Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews – Grégoire Mesnil, Tomas Mikolov, Marc’Aurelio Ranzato, Yoshua Bengio, 2014. [https://arxiv.org/abs/1412.5335?utm_source=chatgpt.com]
2 Fine-tuning BERT with Bidirectional LSTM for Fine-grained Movie Reviews Sentiment Analysis – Gibson Nkhata et al., 2025. [https://arxiv.org/abs/2502.20682?utm_source=chatgpt.com]
3 Enhancing Sentiment Classification with Machine Learning and Combinatorial Fusion – Sean Patten et al., 2025. [https://arxiv.org/abs/2510.27014?utm_source=chatgpt.com]
4 Semantic Sentiment Analysis Based on Probabilistic Graphical Models and RNN – Ukachi Osisiogu, 2020. [https://arxiv.org/abs/2009.00234?utm_source=chatgpt.com]
5 A Comprehensive Benchmarking Pipeline for Transformer-Based Sentiment Analysis using Cross-Validated Metrics – Dodo Zaenal Abidin et al., 2025. [https://www.researchgate.net/publication/381285499_Sentiment_Analysis_of_IMDb_Movie_Reviews?utm_source=chatgpt.com]
6 Sentiment Analysis of IMDB Movie Reviews Using BERT (IARIA conference-ийн paper, 2023). [https://personales.upv.es/thinkmind/dl/conferences/eknow/eknow_2023/eknow_2023_2_30_60011.pdf?utm_source=chatgpt.com]
7 Sentiment Analysis of IMDb Movie Reviews (thesis / Kaggle based) – Domadula 2023. [https://www.diva-portal.org/smash/get/diva2%3A1779708/FULLTEXT02.pdf?utm_source=chatgpt.com]
8 Sentiment Analysis of IMDb Movie Reviews Using LSTM (conference/study) – Saad Tayef (document). [https://www.scribd.com/document/744893212/Sentiment-Analysis-of-IMDb-Movie-Reviews-Using-LSTM?utm_source=chatgpt.com]


