cd /home/jeeves/openmatch/

pip install transformers==4.37.2
pip install deepspeed==0.13.2

echo "transformers, deepspeed setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"
# add pad token
# added to model file

cd Library/GradCache
pip install -e .
cd -


pip install -e .
echo "openmatch setup succeed!"

pip install -r Research/vision/requirements.txt

pip install Research/vision/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

cd Library/pytorch-image-models-0.9.16
pip install -e .
cd -

cd /home/jeeves
mkdir tmp
cd tmp
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/git-lfs-linux-amd64-v3.4.0.tar.gz .
tar zxvf git-lfs-linux-amd64-v3.4.0.tar.gz
cd git-lfs-3.4.0
sudo ./install.sh
