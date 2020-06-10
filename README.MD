Cassandra: Detecting Trojaned Networks from Adversarial Perturbations


Requirements:
1)Python 3.6 (Anaconda3):
  conda create -n envname python==3.7 anaconda
  conda activate envname
2)Pytorch >=1.4:
conda install pytorch torchvision -c pytorch
3)Geffnet and scikit-image
   pip install geffnet scikit-image

Data: download NIST-Round0 data from https://data.nist.gov/od/id/mds2-2175
      and save it in ./data/

Generate meta data for NIST-Round0  models, run
python data_pre.py
output is  data_meta.txt and then split the data into data_meta_tain.txt and data_meta_test.txt

Training
To train the model(s) in the paper, run these commands
1) generate l infinity norm bounded adversarial perturbations for training and test set
and save them to ./Pert/pert_mb/train_inf and ./Pert/pert_mb/test_inf respectively by running:
python pert_gen_mb.py
python pert_gen_mb_test.py 
3) generate l2 norm bounded adversarial perturbations for training and test set
and save them to ./Pert/pert_mb/train_l2 and ./Pert/pert_mb/test_l2 respectively by running:
python pert_gen_l2.py
python pert_gen_l2_test.py 
4) train the model
python train.py
The generated model will be saved in "./models/detector/pretrained/"

Evalution:
To evaluate my pretrained model:
1) download pretrained model from:https://drive.google.com/file/d/1gjrKbiVmYJwNujHN-m9kziy6ViERjezx/view?usp=sharing   and save it to ./models/detector/pretrained/  
2) run 
python eval.py

Results:
model type: mobilenetv3
Evaluation:
Evaluate trojanned model: mobilenetv3_0.95_0.925_mnet.pth
Final Test Accuracy: 0.9500


