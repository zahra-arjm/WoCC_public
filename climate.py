import pandas as pd
import numpy as np


def main():
    

    df = load_dataset_docker()

    
    # add attention-related exclusion columns to the dataframe 
    identify_hastys(df)
    identify_straightliners(df)
    identify_attention_low(df)

    # remove anyone who were exculded by one of the attention-related criteria
    mask = df[
        ['exclude_haste', 'exclude_straightlining', 'exlude_attention']
        ].any(axis='columns')

    # apply mean to the the included individuals
    calculated_value = df[~mask]['prediction'].mean()

    with open('output/result.txt', 'w') as f:
        f.write(str(calculated_value))


def load_dataset_docker():
    """
    A function to load the dataset in the virtual environment
    """
    # load dataset                    
    with open('input/dataset.csv', "r") as f:
        data = pd.read_csv(f)
        return pd.DataFrame(data)



def identify_hastys(df):
    """
    A function to label participants who complete the survey too quickly
    - the threshold is a proportion of the lower quartile of the distribution of speeds
    - i.e. if there paramater is 0.5 anyone who completes in half the time of the respondent
     in the 25th centile for speed is labelled 
    """
    haste_threshold =0.5 #proportion of Q1 speed below which we exclude

    # load duration related variables
    duration_vars = load_duration_vars(df)
    #sum all durations
    sum_duration = df[duration_vars].sum(axis=1)

    #find lower quartile    
    Q1 = sum_duration.quantile(0.25)

    #mark to exclude those who are faster than half the lower quartile of durations
    df['exclude_haste'] = sum_duration < (Q1*haste_threshold)



def identify_straightliners(df):
    """
    A function to identify respondents who answer the same on multiple blocks of questions. 
    This applies to questions in the same scale which all have the same response options
    provided in a matrix.    
    The function looks across all questions to identify where straightlining is possible.    
    The threshold is the number of blocks where straightlining is identified
    """
    straightlining_threshold = 3 #you shouldn't be above this
    ## first identify groups of numbered variables from df.columns which start with the same string
    ## e.g. ius_1 to ius_12
    all_vars = df.columns

    #find all variables which are numbered
    numbered_vars = [var for var in all_vars if ends_with_underscore_integer(var)] 

    #sort by start of string
    numbered_vars.sort()

    #convert to df
    numbered_vars = pd.DataFrame(numbered_vars, columns = ['var'])

    #create group variable by count down the rows incrementing the index each time a var entry ends in _1
    numbered_vars['group'] = (numbered_vars['var'].str[-2:] == '_1').cumsum()


    #iterate through all groups and check for straightlining

    df['straightlining'] = 0

    for group in range(max(numbered_vars['group'])):
        vars = numbered_vars[numbered_vars['group'] == group]['var']
        #find size of group
        if vars.size>4: #this gets rid of some non likert items
            #check for straightlining and increment counter if so
            df['straightlining']=df['straightlining']+(df[vars].nunique(axis=1) == 1).astype(int)
            

    df['exclude_straightlining'] = df['straightlining'] > straightlining_threshold



def identify_attention_low(df):
    attention_threshold = 4 #you need to be above this
    df['exlude_attention'] = ((df['attention_correct'] == False) |
                         (df['attention_study_response'] < attention_threshold))



def load_duration_vars(df):
    """
    load duration related variables in the dataframe
    """
    duration_vars = ['seconds_prediction',
                    'seconds_meta_prediction',
                    'time_consent',
                    'time_introduction',
                    'time_announcement_predictions',
                    'time_prediction_economics',
                    'time_prediction_politics',
                    'time_prediction_sports',
                    'time_prediction_climate',
                    'time_announcement_meta_predictions',
                    'time_meta_prediction_economics',
                    'time_meta_prediction_politics',
                    'time_meta_prediction_sports',
                    'time_meta_prediction_climate',
                    'time_attention_1',
                    'time_attention_2',
                    'time_engagement_predictions',
                    'time_announcement_psychology',
                    'time_announcement_characteristics',
                    'time_external_info_time_pressure',
                    'time_candy_prediction',
                    'time_sdk',
                    'time_fin_lit',
                    'time_math',
                    'time_crt4',
                    'time_b5i',
                    'time_b5o',
                    'time_sop2',
                    'time_oct',
                    'time_tsq',
                    'time_rcs',
                    'time_ius',
                    'time_loc',
                    'time_ih',
                    'time_risk_altruism',
                    'time_knowledge',
                    'time_experience',
                    'time_ses',
                    'time_quant_stat',
                    'time_fan_assets',
                    'time_media_consumption_1',
                    'time_media_consumption_2',
                    'time_spml',
                    'time_demographics_1',
                    'time_demographics_2',]
    #some of these may be missing, so take intersection with variables from df
    duration_vars = list(set(duration_vars).intersection(df.columns))
    return duration_vars



def ends_with_underscore_integer(s):
    """
    Check if the string s ends with an underscore followed by an integer.

    Parameters:
    s (str): The string to be checked.

    Returns:
    bool: True if s ends with an underscore followed by an integer, False otherwise.
    """
    underscore_index = s.rfind('_')  # Find the last underscore

    # Check if underscore is not found or it's at the end of the string
    if underscore_index == -1 or underscore_index == len(s) - 1:
        return False

    # Extract the substring after the underscore
    substring = s[underscore_index + 1:]

    # Check if the substring is an integer
    return substring.isdigit()



# run the main function
if __name__ == "__main__":
    main()