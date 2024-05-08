cd /home/jeeves
mkdir tmp
cd tmp
hadoop fs -get hdfs:///user/tc_agi/user/xubokai/git-lfs-linux-amd64-v3.4.0.tar.gz .
tar zxvf git-lfs-linux-amd64-v3.4.0.tar.gz
cd git-lfs-3.4.0
sudo ./install.sh

cd /home/jeeves/openmatch
sh ./install.sh