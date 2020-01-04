#!/bin/bash -l                                                                                      
#SBATCH --job-name=train20                                                                        
#SBATCH --output=train20_output.txt                                                                        
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
python train_gpu_multibatch.py -train_orig orig_train20.csv -train_tgt trg_train20.csv -valid_orig orig_eval20.csv -valid_tgt trg_eval20.csv -orig_dict en.json -tgt_dict de.json -batch_size 4 -train_steps 12500 -log output -save_model bestmodel -n_warmup_steps 500 -max_seq_len 20
               
