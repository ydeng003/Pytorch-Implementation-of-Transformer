#!/bin/bash -l                                                                                      
#SBATCH --job-name=test20                                                                            
#SBATCH --output=test20_output.txt                                                                        
#SBATCH --ntasks=1                                                                                  
#SBATCH --cpus-per-task=1
#SBATCH -p gpu --gres gpu:1
                                                                          
enable_lmod                                                                                         
module load python
module load cuda
module load pytorch                                                                                  
module load pandas 
module load matplotlib  
module load pytorch-nlp 
module load nltk
python test_gpu.py -orig_dict en.json -tgt_dict de.json -test_orig orig_val20.csv -test_tgt trg_val20.csv -max_seq_len 20 -batch_size 4 -log output
               
