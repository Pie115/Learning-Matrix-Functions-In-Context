import os
import numpy as np
from Generate_Prompt import Generate_Prompt

def run_Vector_P_norm(output_file_name, P, vec_size, min_value, max_value, no_machines, no_examples, no_experiments):
    #Arguments include the following P (what P norm you would like to take), vec_size (The size of the vector you want to input) min_value (The smallest value in your distribution), max_value (the largest value in you distribution), no_machines (The amount of questions you would like to ask prior), no_examples (The number of prior examples per question you would like to show LLM), no_experiments (The number of experiments you would like to run per example, MSE is the average MSE of all experiments per example test

    P_norm = Generate_Prompt()
    df_final_experiment_results, df_average_results  = P_norm.run_multiple_tasks_examples_Vector_P_Norm(p = P, vec_size = 
                                                                                                              vec_size, min_value =
                                                                                                              min_value, max_value = 
                                                                                                              max_value, no_machines =
                                                                                                              no_machines, no_examples = 
                                                                                                              no_examples,
                                                                                                              no_experiments =
                                                                                                              no_experiments)

    df_final_experiment_results.to_csv(f'{output_file_name}_{P}_experiment_results.csv', index=False)
    df_average_results.to_csv(f'{output_file_name}_{P}_average_results.csv', index=False)

def run_Matrix_norm(output_file_name, opp, P, n_size, m_size, min_value, max_value, no_machines, no_examples, no_experiments):
    #Arguments include the following opp (What matrix opperation you would like to run. p for P norm, n for nuclear norm, k for top k singular values), P (what P norm you would like to take if we want to take p norm of matrix, however if you are selecting top k values, this will be your value for k), n_size (the amount of rows you want in your matrix), m_size (the amount of columns you want in your matrix), min_value (The smallest value in your distribution), max_value (the largest value in you distribution), no_machines (The amount of questions you would like to ask prior), no_examples (The number of prior examples per question you would like to show LLM), no_experiments (The number of experiments you would like to run per example, MSE is the average MSE of all experiments per example test
    
    Matrix_norm = Generate_Prompt()
    df_final_experiment_results, df_average_results  = Matrix_norm.run_multiple_tasks_examples_Matrix_Norm(opp = opp,
                                                                                                              p = P, n = n_size,
                                                                                                              m = m_size, min_value =
                                                                                                              min_value, max_value = 
                                                                                                              max_value, no_machines =
                                                                                                              no_machines, no_examples = 
                                                                                                              no_examples,
                                                                                                              no_experiments =
                                                                                                              no_experiments)

    df_final_experiment_results.to_csv(f'{output_file_name}_{opp}_experiment_results.csv', index=False)
    df_average_results.to_csv(f'{output_file_name}_{opp}_average_results.csv', index=False)

#run_Vector_P_norm('LLM_Final_results_P_Norm', P = 0.5, vec_size = 5, min_value = -100, max_value = 100, no_machines = 1, no_examples = 100, no_experiments = 50)

#run_Matrix_norm('LLM_results_Nuclear_Matrix_Norm_50exp_100ex_vec5.txt', opp = 'n', P = 0.5, n_size = 5, m_size = 5, min_value = -100, max_value = 100, no_machines = 1, no_examples = 100, no_experiments = 50)

run_Matrix_norm('LLM_results_Top_K_Matrix_50x50_k3.txt', opp = 'k', P = 3, n_size = 40, m_size = 40, min_value = -100, max_value = 100, no_machines = 1, no_examples = 50, no_experiments = 50)

#if top k svd selected, p is used a the k value
#Adjust opp and p when needed for each experiment