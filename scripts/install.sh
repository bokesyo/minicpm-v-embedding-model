pip install transformers==4.37.2
pip install deepspeed==0.13.2

echo "transformers, deepspeed setup succeed!"

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"

cd Library

cd pytrec_eval
pip install . # here do not use -e .
echo "pytrec_eval setup succeed!"
cd ..

cd sentence-transformers
pip install -e .
echo "sentence-transformers setup succeed!"
cd ..

cd beir
pip install -e .
echo "beir setup succeed!"
cd ..

cd GradCache
pip install -e .
echo "GradCache setup succeed!"
cd ..

cd ..

pip install evaluate
echo "evaluate setup succeed!"



pip install -e .
echo "openmatch setup succeed!"

