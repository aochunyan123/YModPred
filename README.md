# YModPred
# 1. Description

RNA post-transcriptional modifications involve adding chemical groups or changing RNA structure, affecting base pairing, thermal stability, and folding. These changes regulate processes like splicing, translation, localization, and stability. Therefore, accurately predicting RNA modification sites is crucial for understanding how these modifications work. We have developed a novel predictor, YModPred, a deep learning model that predicts ten types of RNA modifications in S. cerevisiae based on RNA sequences. YModPred combines convolution and self-attention mechanisms to enhance prediction accuracy. Comparisons show YModPred outperforms existing methods, with its results validated through visualization and motif analysis. YModPred will aid in advancing research on RNA modification mechanisms.

# 2. Availability
Datasets and source code are available at: https://github.com/aochunyan/YModPred.git

# 3. Requirements
Before running, please make sure the following packages are installed in Python environment:
python==3.8.16
pytorch==2.0
numpy==1.24.3
pandas==2.0.1

# 4. Running
Changing working dir to YModPred, and then running the following command:
python main.py -i test.fasta -o prediction_results.csv
-i: input file in fasta format
-o: output file name
