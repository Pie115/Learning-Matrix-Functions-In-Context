import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import google.generativeai as genai
import time
import os
import re

class Generate_Prompt:

    # Configure Gemini API with API key given by  user
    def __init__(self, key=""):
        self.key = key
        genai.configure(api_key=self.key)

    # Give gemini a prompt
    def generate_gemini(self, prompt):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")  # Preferred Model

        while True:
            try:
                response = model.generate_content([prompt])
                #print(prompt, response.text)
                return response.text.strip()
                # Returns geminis answers as a list.
            except Exception as e:
                if "429" in str(e).lower():
                    print("Rate limit exceeded. Waiting before retrying...")
                    time.sleep(60)
                    # If API limit is reached, try again after a minute
                elif "invalid operation" in str(e).lower():
                    time.sleep(5)
                    print("Invalid operation. Checking candidate safety ratings and retrying...")
                    #time.sleep(5)  # Shorter wait time before retrying
                elif "an internal error" in str(e).lower():
                    print("Internal error. Checking candidate safety ratings and retrying...")
                    #time.sleep(5)  # Shorter wait time before retrying
                else:
                    print("Some other error has occured, retrying...")
                    #time.sleep(5)  # Shorter wait time before retrying

    def generate(self, prompt):
        generated_text = self.generate_gemini(prompt)  # Generate an answer given the prompt.

        return generated_text

    def __generate_Vector_P_Norm(self, p=1, vec_size=5, min_value=-100, max_value=100):  # Generate Lp norm for a given vector, assuming p = 1, and assuming user wants positives only
        np.set_printoptions(threshold=np.inf)

        x_vect = []

        for i in range(vec_size):
            x_vect.append(np.random.uniform(min_value, max_value))  # Random vector is based on the distribution specified by the user.

        y_value = np.power(np.sum(np.power(np.abs(x_vect), p)), (1/p)) # Generate answer for P norm
        return x_vect, y_value

    def __generate_Matrix_P_Norm(self, p=2, n=2, m=2, min_value=-100, max_value=100):
        # Generate a random matrix of size n x m with values between min_value and max_value

        matrix = np.random.uniform(min_value, max_value, (n, m))
        # Compute the p-norm (entrywise) using numpy operations
        p_norm = np.sum(np.abs(matrix) ** p) ** (1 / p)
        
        return matrix, p_norm  # return the matrix and the outputted norm as list to pass through llm

    def __generate_Matrix_Nuclear_Norm(self, n=2, m=2, min_value=-100, max_value=100):
        # Generate a random matrix of size n x m with values between min_value and max_value
        np.set_printoptions(threshold=np.inf)

        matrix = np.random.uniform(min_value, max_value, (n, m))
        # Compute the nuclear norm, which is the sum of the singular values of the matrix
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        nuclear_norm = np.sum(singular_values)
        
        return matrix, nuclear_norm  # return the matrix and the nuclear norm to pass through LLM

    def __generate_Top_k_Singular_Values(self, n=2, m=2, k=2, min_value=-100, max_value=100):
        # Generate a random matrix of size n x m with values between min_value and max_value
        np.set_printoptions(threshold=np.inf) #Makes it so this isn't abreviated

        matrix = np.random.uniform(min_value, max_value, (n, m))
        #matrix = np.random.randint(min_value, max_value, (n, m)) #Keep this on backburner

        # Compute the singular values of the matrix
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        #singular_values = np.round(singular_values, decimals = 2) #Keep this on backburner
        
        top_k_singular_values = singular_values[:k]        

        return matrix, top_k_singular_values # return the matrix and the top k singular values as a vector or np.array


    def __create_prompt_Vector_P_Norm(self, p=1, vec_size=5, min_value=-100, max_value=100, no_machines=1, no_examples=10):
        prompt = ""
        answers = []
        raw_questions = []
        raw_answers = []
        
        if no_machines == 1:
            prompt += "You observe a machine that produces an output y for a given input x:\n"  # Base case prompt if only one question is asked
        else:
            prompt += f"You observe {no_machines} different machines that produce an output y for a given input x.\n"
            # Prompt if multiple questions are aseked

        for machine_count in range(1, no_machines + 1):  # Machines each have the same given number of examples, we want to see if accuracy improves based on examples and questions
            prompt += f"Machine {machine_count}:\n"
            prompt += "If no previous examples, sample y from your prior distribution. But do not give any non-numerical answer, just give the output as a scalar followed by ';'! No words, only numbers and semicolons. Even if you are unsure, try to find some sort of pattern and predict y as well as possible.\n"
            # Prompt inspired from meta in context learning paper

            machine_answers = []

            for example_count in range(no_examples):
                task_question, task_answer = self._Generate_Prompt__generate_Vector_P_Norm(p=p, min_value=min_value,
                                                                                          max_value=max_value, vec_size=vec_size)
                machine_answers.append(task_answer)
                if example_count < no_examples - 1:
                    prompt += f"- Given x={task_question}, y={task_answer}\n"
                else:
                    prompt += f"- Given x={task_question}, Predict y: _____\n"

                raw_questions.append(task_question)
                raw_answers.append(task_answer)
                
            answers.append(machine_answers)
            prompt += "\n"

        return raw_questions, raw_answers, prompt, answers  # Return the final prompt along with the final answers for the prompt, this case p norms
        

    def __create_prompt_Matrix_Norm(self, opp = 'p', p = 1, n=5, m=5, min_value=-100, max_value=100, no_machines=1, no_examples=10):
        prompt = ""
        answers = []
        raw_questions = []
        raw_answers = []
        #in the future, have another parameter that specifies the type of norm we want to perform, p norm, some other norm
        if no_machines == 1:
            prompt += "You observe a machine that produces an output y for a given input x:\n"  # Base case prompt if only one question is asked
        else:
            prompt += f"You observe {no_machines} different machines that produce an output y for a given input x.\n"
            # Prompt if multiple questions are aseked

        for machine_count in range(1, no_machines + 1):  # Machines each have the same given number of examples, we want to see if accuracy improves based on examples and questions
            prompt += f"Machine {machine_count}:\n"
            prompt += "If no previous examples, sample y from your prior distribution. But do not give any non-numerical answer, just give the output as a set of values that matches the length of the previous y's followed by ';'! No words, only numbers and semicolons. Even if you are unsure, try to find some sort of pattern and predict y as well as possible.\n"
            # Prompt inspired from meta in context learning paper

            machine_answers = []

            for example_count in range(no_examples):

            #WE DO THE P NORM FOR NOW of a matrix only, we can do more complex norms soon, add as a parameter
                if(opp == 'p'):
                    task_question, task_answer = self._Generate_Prompt__generate_Matrix_P_Norm(p=p, n=n, m=m, min_value=min_value, max_value=max_value)
                   
                elif(opp == 'n'):
                    task_question, task_answer = self._Generate_Prompt__generate_Matrix_Nuclear_Norm(n=n, m=m, min_value=min_value, max_value=max_value)
                elif(opp == 'k'):
                    task_question, task_answer = self._Generate_Prompt__generate_Top_k_Singular_Values(n=n, m=m, k=p, min_value=min_value, max_value=max_value)
                    #If this one is selected, it would use the p value as the top k instead

                #We would check norm type here
                machine_answers.append(task_answer)
                
                if example_count < no_examples - 1:
                    prompt += f"- Given x={task_question}, y={task_answer}\n"
                else:
                    prompt += f"- Given x={task_question}, Predict y: _____\n"
                    
                raw_questions.append(task_question)
                raw_answers.append(task_answer)
            
            answers.append(machine_answers)
            prompt += "\n"
    
        return raw_questions, raw_answers, prompt, answers  # Return the final prompt along with the final answers for the prompt, this case p norms



    def run_multiple_tasks_examples_Vector_P_Norm(self, p=1, vec_size=5, min_value=-100, max_value=100,
                                                  no_machines=5, no_examples=100, no_experiments=5):
    
        # Initialize error arrays
        mse_per_example = np.zeros(no_examples - 1)
        rmse_per_example = np.zeros(no_examples - 1)
        mae_per_example = np.zeros(no_examples - 1)
    
        # DataFrame to store experiment results
        df_experiment_results = pd.DataFrame(columns=['i', 'j', 'prompt', 'actual_answers', 'predicted_answers', 'raw_questions', 'raw_answers', 'mse', 'rmse', 'mae'])
                                            
        for i in range(2, no_examples + 1):
            j = 0
            while j < no_experiments:
                raw_questions, raw_answers, prompt, actual_answers = self.__create_prompt_Vector_P_Norm(p=p, vec_size=vec_size, min_value=min_value, max_value=max_value, no_machines=no_machines, no_examples=i)
    
                prompt_answers = self.generate_gemini(prompt)
                
                prompt_answers = prompt_answers.split('\n')  # Split the prompts based on \n since LLM was prompted to put this after every answer
                ai_answers_final = []
    
                pattern = re.compile(r'([-+]?\d*\.\d+|\d+);')  # Regular expression to extract numeric answers
                
                for line in prompt_answers:
                    match = pattern.search(line)
                    if match:
                        ai_answers_final.append(float(match.group(1)))
    
                actual_answers_final = [answers[-1] for answers in actual_answers]
                print(j, actual_answers_final, ai_answers_final)
    
                if len(actual_answers_final) == len(ai_answers_final):  # Check if LLM answered in the correct format
                    errors = np.array(actual_answers_final) - np.array(ai_answers_final)
                    squared_errors = np.square(errors)
                    absolute_errors = np.abs(errors)
                    
                    mse = np.mean(squared_errors)  # Mean Squared Error
                    rmse = np.sqrt(mse)            # Root Mean Squared Error
                    mae = np.mean(absolute_errors) # Mean Absolute Error
    
                    # Create a temporary DataFrame for the current experiment
                    temp_df = pd.DataFrame({
                        'i': [i],
                        'j': [j],
                        'prompt': [prompt],
                        'actual_answers': [np.array(actual_answers_final).flatten()],
                        'predicted_answers': [np.array(ai_answers_final).flatten()],
                        'raw_questions': [raw_questions],
                        'raw_answers': [raw_answers],
                        'mse': [mse],
                        'rmse': [rmse],
                        'mae': [mae]
                    })
                    
                    print(temp_df.iloc[0]['actual_answers'], temp_df.iloc[0]['predicted_answers'])
    
                    # Concatenate the temporary DataFrame to the final results DataFrame
                    df_experiment_results = pd.concat([df_experiment_results, temp_df], ignore_index=True)
    
                    # Update error metrics for averaging
                    mse_per_example[i - 2] += mse
                    rmse_per_example[i - 2] += rmse
                    mae_per_example[i - 2] += mae
                    j += 1
                else:
                    print("LLM output incorrect format, skipping and trying again")
    
            print(f'Number of Examples: {i}; MSE, RMSE, MAE: {mse_per_example[i - 2]/no_experiments, rmse_per_example[i - 2]/no_experiments, mae_per_example[i - 2]/no_experiments}')
    
        # Average out error metrics across all experiments
        mse_per_example /= no_experiments
        rmse_per_example /= no_experiments
        mae_per_example /= no_experiments
    
        # Create a summary DataFrame for the average results
        df_average_results = pd.DataFrame({
            'i': range(2, no_examples + 1),
            'average_mse': mse_per_example,
            'average_rmse': rmse_per_example,
            'average_mae': mae_per_example
        })
    
        return df_experiment_results, df_average_results


    def run_multiple_tasks_examples_Matrix_Norm(self, opp='p', p=1, n=5, m=5, min_value=-100, max_value=100,
                                                no_machines=5, no_examples=100, no_experiments=5):
    
        # FINAL ERROR ARRAYS
        mse_per_example = np.zeros(no_examples - 1)
        rmse_per_example = np.zeros(no_examples - 1)
        mae_per_example = np.zeros(no_examples - 1)
        
        # FINAL DATA FRAME PER PROMPT PER QUESTION
        df_experiment_results = pd.DataFrame(columns=['i', 'j', 'prompt', 'actual_answers', 'predicted_answers', 'raw_questions', 'raw_answers', 'mse', 'rmse', 'mae'])
    
        for i in range(2, no_examples + 1):
            j = 0
            while j < no_experiments:
                raw_questions, raw_answers, prompt, actual_answers = self.__create_prompt_Matrix_Norm(opp=opp, p=p, n=n, m=m, 
                                                                                                      min_value=min_value, 
                                                                                                      max_value=max_value, 
                                                                                                      no_machines=no_machines, 
                                                                                                      no_examples=i)
                prompt_answers = self.generate_gemini(prompt)
                prompt_answers = prompt_answers.split('\n')

                ai_answers_final = []
    
                pattern = re.compile(r'([-+]?\d*\.\d+(?:;[-+]?\d*\.\d+)*)')
                
                for line in prompt_answers:
                    match = pattern.search(line)
                    if match:
                        ai_answers_final.extend([float(num) for num in match.group(1).split(';') if num])

                actual_answers_final = [answers[-1] for answers in actual_answers]
    
                # CHECK IF THE AI ANSWERS LENGTH IS THE SAME AS ACTUAL ANSWERS
                if len(np.array(actual_answers_final).flatten()) == len(np.array(ai_answers_final).flatten()):
                    errors = np.array(actual_answers_final) - np.array(ai_answers_final)
                    squared_errors = np.square(errors)
                    absolute_errors = np.abs(errors)
                    
                    mse = np.mean(squared_errors)
                    rmse = np.sqrt(mse)
                    mae = np.mean(absolute_errors)
    
                    # TEMP DF TO CONCAT
                    temp_df = pd.DataFrame({
                        'i': [i],
                        'j': [j],
                        'prompt': [prompt],
                        'actual_answers': [np.array(actual_answers_final).flatten()],
                        'predicted_answers': [np.array(ai_answers_final).flatten()],
                        'raw_questions': [raw_questions],
                        'raw_answers': [raw_answers],
                        'mse': [mse],
                        'rmse': [rmse],
                        'mae': [mae]
                    })
                    print(temp_df.iloc[0]['actual_answers'], temp_df.iloc[0]['predicted_answers'])
    
                    df_experiment_results = pd.concat([df_experiment_results, temp_df], ignore_index=True)
    
                    # CALCULATE FINAL ERRORS ACROSS ALL QUESTIONS
                    mse_per_example[i - 2] += mse
                    rmse_per_example[i - 2] += rmse
                    mae_per_example[i - 2] += mae
                    j += 1
                else:
                    print("LLM output incorrect format, skipping and trying again")
    

            print(f'Number of Examples: {i}; MSE, RMSE, MAE: {mse_per_example[i - 2]/no_experiments, rmse_per_example[i - 2]/no_experiments, mae_per_example[i - 2]/no_experiments}')
    
        # AVERAGE OUT ACROSS ALL EXPERIMENTS
        mse_per_example /= no_experiments
        rmse_per_example /= no_experiments
        mae_per_example /= no_experiments
    
        df_average_results = pd.DataFrame({
            'i': range(2, no_examples + 1),
            'average_mse': mse_per_example,
            'average_rmse': rmse_per_example,
            'average_mae': mae_per_example
        })
    
        return df_experiment_results, df_average_results
