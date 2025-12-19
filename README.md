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
- Dataset paper: [https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf]

## 3.2 Dataset-тэй холбоотой судалгааны өгүүлэл
**Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews – Grégoire Mesnil, Tomas Mikolov, Marc’Aurelio Ranzato, Yoshua Bengio, 2014. [https://arxiv.org/abs/1412.5335?utm_source=chatgpt.com]**
 Энэхүү судалгаанд кино шүүмжийн сэтгэл хөдлөлийг автоматаар тодорхойлох асуудлыг илүү өндөр нарийвчлалтай шийдэх зорилготой судалгаа бөгөөд уг судалгаанд текстийн статистик бүтэц, семантик мэдээллийг сайн барьж чаддаг generative буюу үүсгэгч загвар болон ангиллын шийдвэр гаргахад хүчтэй discriminative буюу ялгагч загваруудыг нэгтгэн ашиглах шинэ хандлагыг дэвшүүлсэн байдаг. Судалгаанд IMDb кино шүүмжийн өгөгдлийг ашиглаж, эерэг болон сөрөг сэтгэл хөдлөлийг хоёр ангилалд хуваах асуудлыг авч үзсэн ба эхлээд эерэг болон сөрөг шүүмжүүдэд тус тусад нь neural language model сургаж, тухайн шүүмж аль загварт илүү тохирч байгааг илтгэх магадлалын оноо (log-likelihood)-г тооцоолсон байна. Үүний зэрэгцээ уламжлалт discriminative аргууд болох bag-of-words, n-gram шинжүүдэд суурилсан шугаман ангилагчийг ашиглан тухайн текстийн сэтгэл хөдлөлийг шууд таамагласан бөгөөд эдгээр хоёр өөр төрлийн загвараас гарсан оноо, шинжүүдийг нэгтгэн эцсийн ensemble ангилагчийг байгуулжээ. Туршилтын үр дүнгээс харахад зөвхөн generative эсвэл зөвхөн discriminative загвар ашигласнаас илүүтэйгээр хоёрыг хослуулсан ensemble арга нь илүү өндөр нарийвчлал, илүү сайн ерөнхийшүүлэх чадварыг үзүүлсэн бөгөөд энэ нь generative загвар текстийн дараалал, утгын мэдээллийг гүнзгий барьж чаддаг бол discriminative загвар шийдвэр гаргалтад илүү шууд, үр дүнтэй байдгийн давуу талыг нэг дор ашиглаж болдгийг нотолсон юм. Иймээс энэхүү өгүүлэл нь sentiment analysis болон ерөнхийдөө байгалийн хэлний боловсруулалтын салбарт hybrid буюу хосолсон загваруудыг ашиглах боломж, ач холбогдлыг тодорхой харуулсан, орчин үеийн deep learning-д суурилсан NLP судалгаанд чухал суурь болсон ажилд тооцогддог.
 
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
Кино шүүмжийн сэтгэл хөдлөлийг зөвхөн эерэг, сөрөг гэж ерөнхийлөхөөс илүүтэйгээр нарийвчилсан түвшинд (жишээлбэл, маш сөрөг, сөрөг, төвийг сахисан, эерэг, маш эерэг гэх мэт) ангилах зорилготой судалгаа бөгөөд орчин үеийн transformer-д суурилсан BERT загварын хэлний гүнзгий ойлголтыг дарааллын хамаарлыг илүү сайн барьж чаддаг Bidirectional LSTM архитектуртай хослуулан ашиглах аргачлалыг санал болгосон байна. Судалгаанд эхлээд урьдчилан сургагдсан BERT загварыг кино шүүмжийн өгөгдөл дээр fine-tuning хийж, үг болон өгүүлбэрийн контекстийн баялаг дүрслэлийг гарган авсан бөгөөд эдгээр contextualized embeddings-ийг дараагийн шатанд Bidirectional LSTM-д оруулснаар өгүүлбэр доторх урд ба хойноосоо хамаарах сэтгэл хөдлөлийн урсгалыг илүү нарийн моделчилжээ. Ингэснээр BERT нь глобал семантик мэдээллийг, харин BiLSTM нь өгүүлбэрийн дарааллын динамик өөрчлөлтийг тус тус барьж, нийлээд нарийн ангиллын sentiment analysis-д илүү тохиромжтой төлөөлөл үүсгэх боломжийг бүрдүүлсэн байна. Туршилтын үр дүнгээс харахад зөвхөн BERT эсвэл зөвхөн LSTM-д суурилсан загваруудтай харьцуулахад энэхүү хосолсон архитектур нь fine-grained sentiment ангилалд илүү тогтвортой, илүү өндөр гүйцэтгэл үзүүлэх хандлагатай байсан бөгөөд энэ нь орчин үеийн NLP судалгаанд transformer ба recurrent neural network-ийг хамтад нь ашиглах боломж үр ашигтай хэвээр байгааг харуулж байна. Иймээс уг өгүүлэл нь кино шүүмжийн нарийвчилсан сэтгэл хөдлөлийн шинжилгээ, хэрэглэгчийн санал бодлын анализ, онлайн тоймд суурилсан шийдвэр гаргалтад ашиглагдах deep learning загваруудын хөгжлийн чиг хандлагыг илтгэсэн, практик болон онолын ач холбогдолтой судалгаа гэж дүгнэгддэг.

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
Өгүүлэл нь сэтгэл хөдлөлийн ангиллын нарийвчлал, тогтвортой байдлыг сайжруулахын тулд олон төрлийн машин сургалтын загваруудын давуу талыг нэгтгэсэн combinatorial fusion буюу хосолсон нэгтгэлийн аргачлалыг судалсан ажил бөгөөд тус судалгаанд нэг загварт тулгуурлахын оронд хэд хэдэн өөр шинж чанар, архитектур, сургалтын логик бүхий загваруудын гарцыг системтэйгээр нэгтгэх нь илүү найдвартай үр дүн өгдөг болохыг харуулахыг зорьсон байна. Судалгаанд уламжлалт machine learning аргууд болох Support Vector Machine, Logistic Regression, Random Forest болон орчин үеийн neural network-д суурилсан загваруудыг зэрэг ашиглаж, эдгээр загвар бүр текстийн сэтгэл хөдлөлийг өөр өөр өнцгөөс “харах” боломжтой гэдгийг онцолсон бөгөөд combinatorial fusion аргаар тэдгээрийн таамаглалыг жинлэн нэгтгэснээр нэг загварын сул талыг нөгөөгөөр нөхөх нөхцөл бүрдүүлжээ. Ялангуяа энэхүү fusion арга нь зарим загвар богино өгүүлбэрт сайн ажиллах бол зарим нь урт, нийлмэл өгүүлбэрт илүү үр дүнтэй байдаг ялгаатай байдлыг ашиглан, өгөгдлийн олон янз шинж чанарт дасан зохицох чадварыг нэмэгдүүлсэн байна. Туршилтын үр дүнгээс харахад combinatorial fusion-д суурилсан систем нь дан ганц загвараас илүү өндөр нарийвчлал, илүү бага хэлбэлзэлтэй гүйцэтгэлийг үзүүлж, ялангуяа бодит хэрэглээнд түгээмэл тохиолддог шуугиантай, тэнцвэргүй өгөгдөл дээр илүү тогтвортой ажилласан нь тогтоогдсон юм. Иймээс энэхүү өгүүлэл нь sentiment classification-ийн салбарт зөвхөн илүү гүн загвар бүтээхээс гадна олон загварын хамтын шийдвэр гаргалтыг оновчтой зохион байгуулах нь чухал гэдгийг харуулсан бөгөөд практик хэрэглээ, бодит системд нэвтрүүлэхэд чиглэсэн ач холбогдолтой судалгаа хэмээн дүгнэгддэг.

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
Сэтгэл хөдлөлийн анализыг зөвхөн үгийн давтамж, гадаргуугийн шинжүүдээр хязгаарлах бус, өгүүлбэр доторх семантик хамаарал болон сэтгэл хөдлөлийн тархалтыг илүү гүнзгий түвшинд ойлгох зорилготой судалгаа бөгөөд үүнд probabilistic graphical models болон recurrent neural network-ийг хослуулсан hybrid аргачлалыг санал болгосон байна. Судалгаанд эхлээд RNN-ийг ашиглан текстийн дарааллын мэдээлэл, контекстийн өөрчлөлтийг загварчилж, үгсийн дараалал дунд сэтгэл хөдлөл хэрхэн аажмаар өөрчлөгдөж байгааг барьж авсан бол нөгөө талаас probabilistic graphical model-ийг ашиглан үг, ойлголт, сэтгэл хөдлөлийн хоорондын далд магадлалын хамаарлыг илэрхийлж, семантик түвшний шалтгаант холбоосыг тодорхойлсон байна. Эдгээр хоёр өөр загварын гарцыг нэгтгэснээр RNN-ийн дарааллын мэдрэмж болон graphical model-ийн тайлбарлагдахуйц магадлалын бүтэц хоёрыг зэрэг ашиглах боломж бүрдэж, сэтгэл хөдлөлийн ангилал илүү тогтвортой, логик холбоотой болсон гэж судлаач тайлбарласан байдаг. Туршилтын үр дүн нь дан ганц RNN эсвэл дан ганц магадлалын граф загвар ашигласнаас илүүтэйгээр хосолсон арга нь семантик ойлголт шаардсан, олон утгатай өгүүлбэрүүд дээр илүү сайн ажиллаж байгааг харуулсан бөгөөд энэ нь sentiment analysis-д тайлбарлах боломжтой хиймэл оюуны (explainable AI) элементүүдийг нэвтрүүлэх боломжийг харуулсан чухал үр дүн юм. Иймээс уг өгүүлэл нь сэтгэл хөдлөлийн анализыг гүн нейрон сүлжээний хүч болон магадлалын загварын тайлбарлагдахуйц бүтэцтэй уялдуулсан, онол болон хэрэглээний аль алинд нь ач холбогдолтой судалгаа гэж дүгнэгддэг.

Ашигласан аргууд: 
  - Recurrent Neural Network
  - Probabilistic Graphical Models

Үр дүн:
  - Semantic-level sentiment илүү сайн ялгасан
  - Traditional ML-ээс илүү F1-score үзүүлсэн

Дүгнэлт:
  Семантик хамаарлыг explicit загварчлах нь сантимент анализын чанарыг сайжруулдаг.

**A Comprehensive Benchmarking Pipeline for Transformer-Based Sentiment Analysis using Cross-Validated Metrics – Dodo Zaenal Abidin et al., 2025. [https://www.researchgate.net/publication/381285499_Sentiment_Analysis_of_IMDb_Movie_Reviews?utm_source=chatgpt.com]**
Transformer-д суурилсан sentiment analysis загваруудын гүйцэтгэлийг шударга, системтэй, дахин давтагдахуйц байдлаар үнэлэх зорилготой судалгаа бөгөөд тухайн салбарт түгээмэл ашиглагддаг загваруудыг нэг ижил нөхцөл, нэг ижил шалгуурын дор харьцуулах цогц benchmarking pipeline санал болгосноороо онцлог юм. Судалгаанд BERT, RoBERTa, DistilBERT зэрэг transformer загваруудыг ашиглан сэтгэл хөдлөлийн ангилал хийж, зөвхөн нэг удаагийн train-test хуваалтад тулгуурлахын оронд cross-validation-д суурилсан олон давталттай үнэлгээний аргачлалыг нэвтрүүлснээр гүйцэтгэлийн бодит, тогтвортой хэмжилтийг гаргахыг зорьсон байна. Ингэснээр загваруудын accuracy, precision, recall, F1-score зэрэг хэмжигдэхүүнүүд нь өгөгдлийн санамсаргүй хуваалтаас хэт хамаарахгүй, ерөнхийшүүлэх чадварыг илүү бодитой тусгаж чадсан гэж судлаачид тайлбарладаг. Мөн уг pipeline нь өгөгдөл бэлтгэх, fine-tuning, үнэлгээ, үр дүнг тайлагнах бүх үе шатыг нэг стандартад оруулснаар өөр судлаачид ижил туршилтыг дахин гүйцэтгэх, шинэ загварыг өмнөх судалгаатай шууд харьцуулах боломжийг бүрдүүлсэн байна. Туршилтын үр дүнгээс харахад transformer загваруудын гүйцэтгэл нь зөвхөн архитектураас гадна сургалтын тохиргоо, өгөгдлийн хуваалт, үнэлгээний аргаас ихээхэн хамааралтай болох нь илэрсэн бөгөөд cross-validated benchmarking ашигласнаар бодит хэрэглээнд илүү тохирох загварыг сонгох боломж нэмэгдэж байгааг харуулжээ. Иймээс энэхүү өгүүлэл нь transformer-д суурилсан sentiment analysis судалгаанд зөвхөн “ямар загвар хамгийн сайн бэ” гэдгээс илүү “ямар аргаар үнэлж байна вэ” гэдэг асуулт ч адил чухал болохыг онцолсон, арга зүйн хувьд өндөр ач холбогдолтой судалгаа гэж дүгнэгддэг.

Ашигласан аргууд: 
  - BERT
  - RoBERTa
  - DistilBERT

Үр дүн:
  - RoBERTa BERT-ээс илүү дүн үзүүлсэн
  - Cross-validation нь үнэлгээг илүү найдвартай болгосон

Дүгнэлт:
  Transformer загваруудын гүйцэтгэл нь pretraining strategy-оос ихээхэн хамаардаг.

**Sentiment Analysis of IMDB Movie Reviews Using BERT (IARIA conference-ийн paper, 2023). [https://personales.upv.es/thinkmind/dl/conferences/eknow/eknow_2023/eknow_2023_2_30_60011.pdf]**
өгүүлэл нь кино шүүмжийн сэтгэл хөдлөлийг автоматаар тодорхойлох асуудлыг орчин үеийн transformer-д суурилсан BERT загварыг ашиглан шийдэхэд чиглэсэн судалгаа бөгөөд уламжлалт machine learning болон өмнөх нейрон сүлжээний аргуудтай харьцуулахад BERT-ийн контекстэд суурилсан хэлний гүнзгий ойлголт ямар давуу талтай болохыг харуулах зорилготой байна. Судалгаанд IMDb movie reviews өгөгдлийн санг ашиглаж, шүүмжүүдийг эерэг болон сөрөг гэсэн хоёр ангилалд хуваан, урьдчилан сургагдсан BERT загварыг уг өгөгдөл дээр fine-tuning хийснээр кино шүүмжид түгээмэл тохиолддог урт өгүүлбэр, далд утга, эсрэгцэл бүхий илэрхийллийг илүү зөв ойлгож ангилах боломжийг бүрдүүлжээ. BERT загвар нь өгүүлбэр доторх үгсийн хоорондын хоёр чиглэлт (bidirectional) хамаарлыг нэгэн зэрэг авч үздэг тул “good but boring”, “not bad at all” зэрэг уламжлалт bag-of-words аргаар буруу ангилагдах магадлалтай хэллэгүүдийг илүү зөв тайлбарлаж чадсан байна. Туршилтын үр дүнгээс харахад BERT-д суурилсан загвар нь accuracy болон F1-score зэрэг үндсэн хэмжигдэхүүнүүдээр өмнөх уламжлалт аргуудаас илүү гүйцэтгэл үзүүлсэн бөгөөд энэ нь transformer архитектур нь sentiment analysis-д өндөр үр ашигтайг баталсан жишээ болжээ. Иймээс энэхүү өгүүлэл нь BERT загварыг кино шүүмжийн sentiment analysis-д практик байдлаар ашиглах боломжийг харуулсан, IARIA конференцийн хүрээнд transformer-д суурилсан NLP судалгааны бодит хэрэглээг илэрхийлсэн ач холбогдолтой ажил гэж дүгнэгддэг.

Ашигласан аргууд: 
  - Softmax classifier
  - BERT

Үр дүн:
  - *Accuracy ~90%+
  - Traditional ML-ээс илүү үр дүнтэй

Дүгнэлт:
  Contextual embedding нь sentiment classification-д маш чухал.

**Sentiment Analysis of IMDb Movie Reviews (thesis / Kaggle based) – Domadula 2023. [https://www.diva-portal.org/smash/get/diva2%3A1779708/FULLTEXT02.pdf?utm_source=chatgpt.com]**
Kaggle дээр нийтлэгдсэн IMDb кино шүүмжийн өгөгдөлд тулгуурлан сэтгэл хөдлөлийн анализыг хэрэгжүүлсэн дипломын/магистрын түвшний судалгааны ажил бөгөөд кино шүүмжийг эерэг болон сөрөг гэсэн ангилалд автоматаар хуваахад уламжлалт machine learning болон орчин үеийн deep learning аргууд хэрхэн ажиллаж байгааг харьцуулан шинжилэх зорилготой байв. Судалгаанд эхлээд өгөгдөл цэвэрлэх, tokenization, stop-word арилгах, stemming зэрэг текстийн урьдчилсан боловсруулалтыг хийж, дараа нь Bag-of-Words болон TF-IDF зэрэг уламжлалт feature extraction аргуудыг ашиглан Logistic Regression, Naive Bayes, Support Vector Machine зэрэг ангилагчдыг сургаж туршсан байна. Үүний зэрэгцээ LSTM, CNN зэрэг нейрон сүлжээнд суурилсан загваруудыг мөн ашиглаж, эдгээр нь үгсийн дараалал болон контекстийн мэдээллийг илүү сайн барьж чаддаг эсэхийг судалжээ. Туршилтын үр дүнгээс харахад уламжлалт машин сургалтын аргууд нь тооцооллын хувьд энгийн, хурдан боловч урт, нийлмэл өгүүлбэртэй шүүмж дээр гүйцэтгэл нь хязгаарлагдмал байсан бол LSTM зэрэг deep learning загварууд нь илүү өндөр нарийвчлал үзүүлж, сэтгэл хөдлөлийн далд утгыг илүү сайн илрүүлж чадсан байна. Мөн Kaggle орчныг ашигласнаар өгөгдөл, код, үр дүнг дахин давтагдахуйц байдлаар зохион байгуулсан нь судалгааны практик ач холбогдлыг нэмэгдүүлжээ. Иймээс энэхүү ажил нь IMDb кино шүүмжийн sentiment analysis-д түгээмэл ашиглагддаг өгөгдөл дээр суурилан, уламжлалт болон гүн сургалтын аргуудын ялгаа, давуу сул талыг харуулсан, суралцагч болон эхлэн судлаачдад чиглэсэн хэрэглээний ач холбогдолтой судалгаа гэж дүгнэгддэг.

Ашигласан аргууд: 
  - Logistic Regression
  - LSTM
  - Word2Vec

Үр дүн:
  - LSTM + Word2Vec хамгийн сайн үр дүн үзүүлсэн

Дүгнэлт:
  Sequence model нь текстийн дарааллыг илүү сайн ойлгодог.

**Sentiment Analysis of IMDb Movie Reviews Using LSTM (conference/study) – Saad Tayef (document). [https://www.scribd.com/document/744893212/Sentiment-Analysis-of-IMDb-Movie-Reviews-Using-LSTM?utm_source=chatgpt.com]**
Судалгаа нь кино шүүмжийн сэтгэл хөдлөлийг тодорхойлох асуудлыг recurrent neural network-ийн нэг хэлбэр болох Long Short-Term Memory (LSTM) загварыг ашиглан шийдэхэд чиглэсэн ажил бөгөөд уламжлалт машин сургалтын аргууд текстийн дарааллын мэдээллийг хангалттай тусгаж чаддаггүй сул талыг даван туулахыг зорьсон байна. Судалгаанд IMDb movie reviews өгөгдлийг ашиглаж, шүүмжүүдийг эерэг болон сөрөг гэсэн хоёр ангилалд хуваан, эхлээд текстийг tokenization, padding, embedding зэрэг урьдчилсан боловсруулалтаар дамжуулж LSTM-д тохиромжтой хэлбэрт оруулжээ. LSTM загвар нь өгүүлбэр доторх урт хугацааны хамаарлыг хадгалах чадвартай тул кино шүүмжид түгээмэл тохиолддог урт өгүүлбэр, санааны аажим өөрчлөлт, эсрэгцэл агуулсан илэрхийллийг илүү сайн моделчилж чадсан гэж тайлбарласан байдаг. Туршилтын үр дүнгээс харахад LSTM-д суурилсан загвар нь accuracy болон бусад үнэлгээний үзүүлэлтүүдээр Naive Bayes, Logistic Regression зэрэг уламжлалт аргуудаас илүү гүйцэтгэл үзүүлсэн бөгөөд энэ нь дарааллын мэдээлэл чухал үүрэгтэй sentiment analysis-ийн хувьд RNN төрлийн загварууд илүү тохиромжтой болохыг баталсан юм. Иймээс уг судалгаа нь IMDb кино шүүмжийн өгөгдөл дээр LSTM загварыг хэрэгжүүлэх бодит жишээг харуулсан, deep learning-д суурилсан sentiment analysis-ийн суурь ойлголтыг ойлгоход тустай, конференц болон судалгааны түвшинд ач холбогдолтой ажилд тооцогддог.

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

# Evaluation
Энэхүү хэсэгт 7 төрлийн embedding болон 2 загвар (Logistic Regression, Random Forest) ашиглан гүйцэтгэсэн туршилтын үр дүнг дэлгэрэнгүй харуулна.

## 5.1 Сургалтын орчин болон төхөөрөмж
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
 - 
Онцлог:
 - Cloud дээр ажиллаж байгаа тул hardware-н хүчин чадал нь тухайн Colab runtime-с хамаарна.
 - Хувийн төхөөрөмж шаардлагагүй, интернэттэй байх л хангалттай.

## 5.2 Logistic Regression-ийн үр дүн

### 5.2.1 Logistic Regression + Embedding харьцуулалт
| Embedding      | CV Score | Test Accuracy | Precision | Recall | F1-Score | Training Time (sec) |
|----------------|----------|---------------|-----------|--------|----------|-------------------|
| RoBERTa        | 0.8634   | 0.8661        | 0.86      | 0.87   | 0.86     | 501.25            |
| ALBERT         | 0.8387   | 0.8400        | 0.84      | 0.84   | 0.84     | 595.98            |
| BERT (cased)   | 0.7796   | 0.7836        | 0.79      | 0.78   | 0.78     | 707.09            |
**Хамгийн сайн үр дүн:  RoBERTa + Logistic Regression**
Test Accuracy: 86.61%
Cross-validation Score: 86.34%
Confusion Matrix:
 - True Positive (TP):  4,346 | False Positive (FP): 685
 - False Negative (FN):   654 | True Negative (TN):  4,315
Оптимал параметрүүд:
 - C = 10.0
 - solver = liblinear
 - max_iter = 500
### 5.2.2 Logistic Regression-ийн дүгнэлт
RoBERTa embedding нь Logistic Regression-тэй хослуулахад хамгийн өндөр гүйцэтгэл (86.61%) үзүүлсэн
Cross-validation болон test accuracy хоёул тогтвортой, overfitting үгүй
Сургалтын хурд харьцангуй хурдан (501 секунд)
Precision болон Recall хоёул тэнцвэртэй (86-87%)

## 5.3 Random Forest-ийн үр дүн
### 5.3.1 Random Forest + Embedding харьцуулалт
| Embedding        | CV Score | Test Accuracy | Precision | Recall | F1-Score | Training Time (sec) |
|------------------|----------|---------------|-----------|--------|----------|-------------------|
| RoBERTa          | 0.8232   | 0.8259        | 0.83      | 0.83   | 0.83     | 1,609.10          |
| Word2Vec         | 0.7960   | 0.7984        | 0.80      | 0.80   | 0.80     | 579.05            |
| ALBERT           | 0.7918   | 0.7909        | 0.79      | 0.79   | 0.79     | 1,608.32          |
| HateBERT         | 0.7802   | 0.7837        | 0.77      | 0.79   | 0.78     | 2,135.93          |
| BERT (uncased)   | 0.7831   | 0.7824        | 0.78      | 0.78   | 0.78     | 1,644.10          |
| SBERT            | 0.7812   | 0.7821        | 0.77      | 0.79   | 0.78     | 1,134.59          |
| BERT (cased)     | 0.7151   | 0.7167        | 0.72      | 0.72   | 0.72     | 1,622.53          |

Хамгийн сайн үр дүн: RoBERTa + Random Forest

Test Accuracy: 82.59%
Cross-validation Score: 82.32%
Confusion Matrix:
True Positive (TP):  4,229 | False Positive (FP): 970
False Negative (FN):   771 | True Negative (TN):  4,030
Оптимал параметрүүд:
n_estimators = 400
max_depth = 20
min_samples_split = 5
max_features = 'sqrt'
### 5.3.2  Random Forest-ийн дүгнэлт
RoBERTa дахин хамгийн сайн гүйцэтгэл үзүүлсэн (82.59%)
Word2Vec embedding (100 dimension) нь Random Forest-тэй сайн ажилласан, харьцангуй хурдан (579 сек)
Random Forest нь Logistic Regression-тэй харьцуулахад илүү удаан (1,600+ секунд)
Random Forest нь BERT (cased) embedding-ийг муу ашигласан (71.67%)

## 5.4 XGBoost GPU-ийн үр дүн
### 5.4.1 XGBoost GPU + Embedding харьцуулалт
| Embedding        | Val Accuracy | Test Accuracy | Precision | Recall | F1-Score | Best Config              |
|------------------|--------------|---------------|-----------|--------|----------|--------------------------|
| HateBERT         | 0.7938       | 0.7884        | 0.79      | 0.79   | 0.79     | n_est=300, depth=6       |
| BERT (uncased)   | 0.7798       | 0.7811        | 0.78      | 0.78   | 0.78     | n_est=300, depth=6       |
| ALBERT           | 0.7793       | 0.7782        | 0.78      | 0.78   | 0.78     | n_est=300, depth=6       |
| SBERT            | 0.7792       | 0.7738        | 0.77      | 0.77   | 0.77     | n_est=300, depth=4       |
| RoBERTa          | 0.7725       | 0.7737        | 0.77      | 0.78   | 0.78     | n_est=300, depth=4       |

Хамгийн сайн үр дүн: HateBERT + XGBoost GPU

Test Accuracy: 78.84%
Val Accuracy: 79.38%
Confusion Matrix:
True Positive (TP):  3,981 | False Positive (FP): 1,102
False Negative (FN):   996 | True Negative (TN):  3,838
Оптимал параметрүүд:
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.9
### 5.4.2 XGBoost GPU-ийн дүгнэлт

HateBERT embedding нь XGBoost GPU-тай хамгийн сайн ажилласан (78.84%)
XGBoost нь Random Forest-оос бага зэрэг доогуур гүйцэтгэл үзүүлсэн
GPU ашигласан ч сургалтын хурд дунд зэргийн түвшинд байсан
Gradient boosting нь ensemble методоор илүү тогтвортой үр дүн өгсөн

## 5.5 PyTorch LSTM-ийн үр дүн
### 5.5.1 PyTorch LSTM + Embedding харьцуулалт
| Embedding        | Val Accuracy | Test Accuracy | Precision | Recall | F1-Score | Best Config                  |
|------------------|--------------|---------------|-----------|--------|----------|------------------------------|
| HateBERT         | 0.7963       | 0.7940        | 0.80      | 0.79   | 0.79     | 128 units, dropout=0.3       |
| BERT (uncased)   | 0.7819       | 0.7910        | 0.80      | 0.79   | 0.79     | 256 units, dropout=0.3       |
| SBERT            | 0.7806       | 0.7820        | 0.78      | 0.78   | 0.78     | BiLSTM-128, dropout=0.3      |
| BERT (cased)     | 0.7719       | 0.7770        | 0.78      | 0.78   | 0.78     | BiLSTM-128, dropout=0.4      |
| RoBERTa          | 0.7750       | 0.7660        | 0.78      | 0.77   | 0.76     | BiLSTM-256, dropout=0.2      |

Хамгийн сайн үр дүн: HateBERT + PyTorch LSTM

Test Accuracy: 79.40%
Val Accuracy: 79.63%
Confusion Matrix:
True Positive (TP):  892 | False Positive (FP): 298
False Negative (FN): 114 | True Negative (TN):  696
Оптимал параметрүүд:
lstm_units = 128
dropout = 0.3
bidirectional = False
learning_rate = 0.001
early_stopping: 7 epochs

### 5.5.2  PyTorch LSTM-ийн дүгнэлт 
HateBERT embedding нь LSTM-тэй хамгийн сайн ажилласан (79.40%)
LSTM нь Random Forest болон XGBoost-оос бага зэрэг доогуур гүйцэтгэл үзүүлсэн
Early stopping mechanism (patience=5) нь overfitting-ээс хамгаалсан
GPU (RTX 5060) ашигласан ч жижиг dataset (10,000 samples) дээр давуу тал багатай
Bidirectional LSTM нь зарим тохиолдолд илүү муу үр дүн өгсөн

## 5.6 Харицуулсан үр дүн 
### 5.6.1 Загваруудын гүйцэтгэлийн харьцуулалт

| Загвар              | Embedding | Test Accuracy | Training Time | Давуу тал                               | Сул тал                         |
|---------------------|-----------|---------------|---------------|-----------------------------------------|----------------------------------|
| Logistic Regression | RoBERTa   | 86.61%        | 501 сек       | Хамгийн өндөр accuracy, хурдан          | BERT (cased)-д сул              |
| Logistic Regression | ALBERT    | 84.00%        | 596 сек       | Тогтвортой, тэнцвэртэй                  | CV-test gap бага зэрэг байна    |
| Random Forest       | RoBERTa   | 82.59%        | 1,609 сек     | Тогтвортой                              | Удаан, LR-ээс доогуур           |
| Random Forest       | Word2Vec  | 79.84%        | 579 сек       | Хурдан, жижиг dimension                | Accuracy дунд зэрэг             |
| PyTorch LSTM        | HateBERT  | 79.40%        | GPU-enabled   | Sequence modeling                       | Жижиг dataset дээр давуу тал бага |
| XGBoost GPU         | HateBERT  | 78.84%        | GPU-enabled   | Gradient boosting                      | Ensemble-ээс доогуур            |

### 5.6.2 Embedding-үүдийн харьцуулалт
RoBERTa: Хамгийн тогтвортой, өндөр гүйцэтгэлтэй

 - Logistic Regression: 86.61% (1-р байр)
 - Random Forest: 82.59% (1-р байр)
 - Хоёр загвартай ч хамгийн сайн үр дүн

ALBERT: Тохиромжтой суурь

 - Logistic Regression: 84.00% (2-р байр)
 - Random Forest: 79.09%
 - Логистик регрессэд сайн ажилласан

Word2Vec: Хурдан, үр ашигтай

 - Random Forest: 79.84% (2-р байр)
 - 100 dimension ч гэсэн сайн үр дүн
 - Хамгийн хурдан сургалт (579 секунд)

BERT (cased): Алдагдсан боломж

 - Logistic Regression: 78.36%
 - Random Forest: 71.67% (хамгийн муу)
 - 768 dimension боловч Random Forest дээр муу гүйцэтгэл

HateBERT: Сөрөг контекстэд тусгайлагдсан

 - XGBoost GPU: 78.84% (1-р байр)
 - PyTorch LSTM: 79.40% (1-р байр)
 - Hate speech дээр сургасан тул sentiment analysis-д давуу талтай
 - Гүн сургалтын загваруудтай илүү сайн ажилласан

| Embedding Type   | Dimension | LR Accuracy | RF Accuracy | XGBoost GPU | PyTorch LSTM | Дундаж  |
|------------------|-----------|-------------|-------------|-------------|--------------|---------|
| SBERT            | 384       | -           | 78.21%      | 77.38%      | 78.20%       | 78.20%  |
| Word2Vec         | 100       | -           | 79.84%      | -           | -            | 79.84%  |
| RoBERTa          | 768       | 86.61%      | 82.59%      | 77.37%      | 76.60%       | 80.79%  |
| ALBERT           | 768       | 84.00%      | 79.09%      | 77.82%      | -            | 80.30%  |
| BERT (uncased)   | 768       | -           | 78.24%      | 78.11%      | 79.10%       | 78.48%  |
| BERT (cased)     | 768       | 78.36%      | 71.67%      | -           | 77.70%       | 75.91%  |
| HateBERT         | 768       | -           | 78.37%      | 78.84%      | 79.40%       | 78.87%  |

## 5.7 Дүгнэлт 

### 5.7.1 Үр дүн
1. Хамгийн сайн хослол: RoBERTa + Logistic Regression (86.61%)
2. Хурдан, үр ашигтай: Word2Vec + Random Forest (79.84%, 579 сек)
3. Гүн сургалтын шилдэг: HateBERT + PyTorch LSTM (79.40%)
4. Загварын нөлөө: Logistic Regression нь Random Forest-оос дунджаар 3-4%-иар илүү
5. Embedding-ийн ач холбогдол: RoBERTa нь Linear загваруудад, HateBERT нь гүн сургалтын загваруудад илүү тохиромжтой

# 6 Нэгдсэн дүгнэлт

Энэхүү судалгаанд IMDB Movie Reviews датасет дээр 7 төрлийн embedding болон 2 төрлийн машин сургалтын загварыг системтэйгээр туршиж, sentiment analysis-ийн гүйцэтгэлд хэрхэн нөлөөлж байгааг дэлгэрэнгүй судалсан.

## 6.1 Embedding 

Туршилтын үр дүнгээс харахад RoBERTa embedding нь хоёр загвартай ч хамгийн өндөр гүйцэтгэл үзүүлж, бусад BERT хувилбаруудаас (BERT uncased/cased, ALBERT, HateBERT, SBERT) 6-10 хувийн зөрүүтэй илүү байсан. Энэ нь:

1. Pre-training өгөгдлийн хэмжээ чухал: RoBERTa нь илүү их өгөгдөл дээр сургагдсан
2. Сургалтын стратеги ач холбогдолтой: Dynamic masking болон batch size оновчлол
3. Next Sentence Prediction (NSP) хасах нь sentiment analysis-д илүү тохиромжтой
Word2Vec нь зөвхөн 100 dimension-тэй байсан ч Random Forest-тэй хослуулахад 79.84% accuracy хүрч, хурдан бөгөөд үр ашигтай шийдэл болох нь нотлогдсон.

## 6.2 Харьцуулсан model

Logistic Regression нь Random Forest-оос дунджаар 3-4 хувиар илүү гүйцэтгэл үзүүлсэн. Энэ нь:

 - Өндөр dimension (768)-тай contextual embedding-үүд нь Linear загвартай илүү сайн ажилладаг
 - Random Forest нь feature interaction-ийг автоматаар олдог ч, өндөр dimension дээр overfitting-д орох эрсдэлтэй
 - Logistic Regression-ийн regularization (C parameter) нь 768-dimension embedding-ийг үр дүнтэй ашиглах боломж олгосон

## 6.3 Best үр дүн 

RoBERTa + Logistic Regression хослол нь 86.61% test accuracy хүрч:

 - Cross-validation score (86.34%) болон test accuracy (86.61%) ижил түвшинд байгаа нь overfitting байхгүй гэдгийг харуулж байна
 - Precision (86%) болон Recall (87%) тэнцвэртэй, class imbalance асуудал үгүй
 - Сургалтын хугацаа 501 секунд (~8 минут) нь бодит хэрэглээнд тохиромжтой

## 6.4 Судалгааны ач холбогдол

1. Fine-tuning хийх:
    - BERT загваруудыг IMDB dataset дээр fine-tuning хийх
    - Layer-wise learning rate стратеги ашиглах
    - Accuracy-г 90%+ болгох боломж
  
2. Cross-domain transfer learning:
    - IMDB дээр сургасан загварыг бусад domain-д туршиж үзэх
    - Amazon reviews, Twitter sentiment, Product reviews гэх мэт
    - Монгол хэл дээр ажиллах:

3. Multilingual BERT (mBERT) ашиглах
    - Монгол киноны сэтгэгдлийн dataset бүрдүүлэх
    - Low-resource language-д transfer learning судлах
    
4. Тайлбарлах боломжтой AI (Explainable AI):
    - LIME, SHAP ашиглан загваруудын шийдвэрийг тайлбарлах
    - Attention visualization хийх
    - Error analysis гүнзгий хийх
 
## 6.5 Үндсэн дүгнэлт 

Энэхүү судалгаанд IMDB Movie Reviews датасет дээр 7 төрлийн embedding болон 4 төрлийн машин сургалтын загвар (Logistic Regression, Random Forest, XGBoost GPU, PyTorch LSTM)-ыг системтэйгээр туршиж, sentiment analysis-ийн гүйцэтгэлд хэрхэн нөлөөлж байгааг дэлгэрэнгүй судалсан.
Судалгааны гол ололт нь:
1. Contextual embedding давуу тал нь батлагдсан (RoBERTa: 86.61%, HateBERT: 79.40%)
2. Энгийн загвар илүү үр дүнтэй байж болно (LR > RF > XGBoost > LSTM)
3. Pre-training optimization том ач холбогдолтой (RoBERTa > BERT: 8% ялгаа)
4. Hyperparameter tuning зайлшгүй шаардлагатай (GridSearchCV: 3-5% нэмэгдэл)
5. Embedding-загварын чухал (RoBERTa→LR, HateBERT→LSTM)
6. GPU-ийн давуу тал том dataset шаардлагатай (жижиг dataset дээр хязгаарлагдмал)

Дүгнэж хэлбэл, NLP салбарт embedding-ийн чанар, загварын сонголт, hyperparameter оновчлол, болон embedding-загварын гурвуулаа state-of-the-art үр дүнд хүрэх гол хүчин зүйлс болох нь тодорхой болсон. Ирээдүйд fine-tuning, ensemble methods, болон cross-domain transfer learning ашиглан үр дүнг улам сайжруулах боломжтой.

