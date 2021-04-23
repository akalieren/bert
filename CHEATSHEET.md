# PRETRAINING BERT

I have used bunch of sources but mostly follows [Google's Original TF Implementation](https://github.com/google-research/bert) and [Stefan's BERTurk repository](https://github.com/stefan-it/turkish-bert). However, I encountered problems and revise the script to train BERT.

I have changed small things in Google's Implementation (adding progress bar / debugging problems about Turkish characters etc.) so feel free to use [mine](https://github.com/akalieren/bert) if you want. However, there will no differences during implementation, only differences are new helper scripts and Turkish characters.  I also added `create_vocabs.py`which initialize tokenizer-vocab.txt from from input files.
  
## PACKAGE VERSIONS  
These versions are used to train BERT, up-to-date version of TensorFlow and tokenizer not stable with the script. (e.g `tokenizer.save()`save json file instead of vocabs  in newer versions)
- tensorflow\==1.15.3
- tokenizer\==0.4.2  

## Google Cloud SDK

Google provide powerful apis to fasten this process for us. For the newbies, $300 credits are provided initially. So it is enough for us in this case. We will use Virtual Machine, TPUs Computing and Cloud Storage. 

### Cloud Storage
Since BERT models needs huge data to be trained, we need to store and stream data. 

I have used bunch of corpus including private ones. But if you just want to explore BERT, you may use [OSCAR](https://oscar-corpus.com) corpus which have 27G Turkish raw text. I have used Google Colab to download and copy data to the Google Cloud Storage, because it is fast. Then I copied it to directory where I will run preprocessing script.

If you want, you may use it from this [link](https://colab.research.google.com/drive/1dx9JO_R--0zv_O9nBLNtiC6pu9yEWMkp?usp=sharing)

or you may directly download using HuggingFace Datasets library.
```python
#pip3 install datasets
#pip3 install packaging
from datasets import load_dataset  
dataset = load_dataset('oscar', 'unshuffled_deduplicated_tr')  
dataset.save_to_disk('./')
```

Login your Google Account to [Cloud Console](console.cloud.google.com) 

**Authenticating the Google Cloud SDK**
```shell
gcloud auth login
```
We need to set default region and zone. If you have applied [TFRC](https://www.tensorflow.org/tfrc/), you will have exact config zone here. This config will be used in our all gcloud commands

**Creating a New Project**
Creating a new project. Our PROJECT_ID must be unique and contains lowered alphabets
```
gcloud projects create ${PROJECT_ID}
```
And then init gcloud and choose proper configs
```
gcloud init
```
Then we need to config region and zone for this project

**Project Config**
```
gcloud compute project-info add-metadata --metadata google-compute-default-region=europe-west4,google-compute-default-zone=europe-west4-a
```
If you encounter an error indicating you are not enabled Compute Engine for this project. You should go the the Console and enable Compute Engine billing. For details [click](https://cloud.google.com/apis/docs/getting-started#enabling_apis).

**Now we can create our TPU Engine and Virtual Machine to start our training!**

**Creating a Cloud Bucket**
```
gsutil mb gs://bertpretraining
```
**NOTE:** I recommended to continue next steps after creating pertaining data. This step could takes 1-2 days, thus no need to occupy Computing Engine. Sometimes it could be busy and hard to reserve.

**Creating a TPU Node**
```
gcloud compute tpus create bert --zone='europe-west4-a'  --accelerator-type=v3-8  --version=1.15.3
```
**Creating a VM**
```
gcloud beta compute --project=${PROJECT_ID} instances create bertvm --zone=europe-west4-a --machine-type=n1-standard-16 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --scopes=cloud-platform --image=debian-9-tf-1-15-4-v20201019 --image-project=ml-images --boot-disk-size=500GB --boot-disk-type=pd-balanced --boot-disk-device-name=bertvm --reservation-affinity=any
```
We can connect to server which we create above with ssh like that. We state our Virtual Machine name as a *bertvm* and it is related with the our project id.
```
gcloud compute ssh betvm
```

## TOKENIZER  
We will use pretrained tokenizer with just constructing a vocab file from our dataset.  WordPiece Tokenizer is used in BERT models (also Electra). More information about tokenizers can be obtained from [docs](https://huggingface.co/transformers/tokenizer_summary.html).
   
   I added a snipped creating vocab.txt. Check tokenizers version before running this script. It should be 0.4.2.
```shell  
 git clone https://github.com/google-research/bert.git
 ```  
**Cased Model**
```shell  
 python3 bert/create_vocabs.py --input_file ./turkishcorpus/*  --output_dir ./cased
 ```  
**Uncased Model** add `--uncased`
```shell  
 python3 bert/create_vocabs.py --input_file ./turkishcorpus/* --output_dir ./uncased --uncased
 ```  
 **NOTE:**  In lodoos's implementation and the other's in internet, they add strip_accents=True to initialization of tokenizers.  
This cause problems in Turkish due to Turkish characters. It is about BERT normalizer and unicode decoding.

**Details:** https://huggingface.co/docs/tokenizers/python/latest/api/reference.html?highlight=strip_accents 

Problems starting at strip_accent parameter and Turkish Uppercase I character. strip_accent is called at `BasicTokenizer` of Official Script. It basically decompose canonical characters. So in regular alphabet there isn't "Ş, Ö, Ü, İ" so strip accents makes them S, O, U, I. Because in unicode decoding that are defining from their decomposed from.
More information can be obtained from [documentation](https://www.unicode.org/reports/tr15/tr15-18.html#Decomposition).

What I did here is, adding `strip_accents` as a parameter (default false) and restrict tokenizer to stripping accent with lowering characters. Because in official repository, `Basic Tokenizer` apply strip_accents with lowering cases. If our model is cased, if`do_lower_case=False`, accents are stripped if parameter is `True`.
  
## PREPARE DATA TO PRETRAINING  

If you are running on huge dataset, you may want to split dataset to small chards, because sizes will be increases after this step done and it will be more hard to copy it to **Google Buckets**.
```shell  
split -C 650M ~/corpus/tr_part_1.txt tr_part_1-
 ```  
if you want to parallelize, I added a script in repository called `parallel_preprocessor.py`. I added sample command for for 3 workers.
```
python3 bert/parallel_preprocessor.py --script bert/create_pretraining_data.py --corpus {CORPUS_DIR} --output ./tfrecords --max_seq_length 512 --num_thread 3  --vocab_file {VOCAB_DIR} --uncased
```
  or directly run following commands with 1 workers. No parallelization
```shell  
 python3 bert/create_pretraining_data.py --input_file ./corpus/% --output_file tfrecords/%.tfrecord --vocab_file ../vocabs/uncased/vocab.txt --do_lower_case=True -max_seq_length=512 --max_predictions_per_seq=75 --masked_lm_prob=0.15 --random_seed=43 --dupe_factor=5 
 ```  
This process takes time depends on your data size. After this step, you will have ready-to-train for the TPU. But it is in your local directory. So you need to copy them to **Google Buckets** where you stream data to your model.

```shell
gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M cp -r tfrecords/*  gs://turkishcorpus/tfrecords/
```
This will copy your data to the Buckets. If the size is more than 150mb, it will parallelize process automatically. 

Finally, we need to connect **Google Cloud Virtual Machine** to start pertaining. If you did not create TPU and VM, turn back to the beginning and continue steps. 
```
gcloud compute ssh bertvm
```
Now you are in Google Server and ready to run pretraining. You need to git clone again to run pertaining btw. I did not change anything in pertaining script. 

**Config**
Be sure your `vocab_size`  is same with you create at the beginning. 
```
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 32000
}
```
```shell  
python3 bert/run_pretraining.py --input_file=gs://turkishcorpus/bert_uncased/tfrecords/*.tfrecord --output_dir=gs://turkishcorpus/bert_uncased --bert_config_file={CONFIG_FILE} --max_seq_length=512 --max_predictions_per_seq=75 --do_train=True --train_batch_size=128 --num_train_steps=3000000 --learning_rate=1e-4 --save_checkpoints_steps=100000 --keep_checkpoint_max=20 --use_tpu=True --tpu_name=bert --num_tpu_cores=8  
 ```  
If you encounter problem about permission. You need to give access to your TPUs to the Cloud Bucket. To do that, Yo need to go **IAM&Admin** at Cloud Console and give access to your compute node to the Cloud Storage. Detailed information can be obtained from [documentation](https://cloud.google.com/storage/docs/access-control/iam-permissions). 

And your models will be trained in a few days depends on your data size. 

## POSSIBLE ERRORS    
**About Mismatched Parameters**    
Our parameters should exactly same in all steps. For example, it `max_seq_length` is not equal in `run_pretraining.py` and `create_pretraining_data.py` it will collapse our run time. This same with most parameters such as `max_predictions_per_seq`, `vocab_size` etc.
- https://github.com/tensorflow/tensorflow/issues/36136 

**About Permissions in Cloud**    
There is also possible case where our Virtual Machine or Compute Engine can't access file. Cloud SDK automatically warn us about permissions however, I recommended to check this permissons from Console IAM.

**Wrong Path**    
Another case is stating wrong path when training model. Since path is stated with glob, there can be case where unwanted files tried to be parsed by script. In this case, model will try to parse this file again and again which sometimes cause loop. So I recommended to check this from logs or directly Python Console using `tf.gfile.Glob(PATH)`. It sounds like dummy error but I see lots of people who encounter this in issues. I also did that.

