import pandas as pd
import numpy as np


def main():
    

    df = load_dataset_docker()

    
    # add attention-related exclusion columns to the dataframe 
    identify_hastys(df)
    identify_straightliners(df)
    identify_attention_low(df)

    # add performance-related exclusion columns to the dataframe; some of the algorithms use it
    identify_worst_performers(df)

    # remove anyone who were exculded by one of the attention-related criteria
    mask = df[
        ['exclude_haste', 'exclude_straightlining', 'exlude_attention']
        ].any(axis='columns')

    # update df
    df = df[~mask]

    algorithms = [mean_remove_worst, 
                  median_remove_worst,
                  geo_mean,
                  diversify_responses,
                  top_k_mean,
                  CWM]

    # initiate a dictionary to keep predicted values by each algorithm
    predictions = {algorithm: [] for algorithm in algorithms}
    # predict each month with the algorithms
    for algorithm in algorithms:
        for month in [1, 2]:
            predictions[algorithm].append(algorithm(df, month=month))

    # extract true values for both months
    true_values = [df['true_value_m1'].to_list()[0], df['true_value_m2'].to_list()[0]]

    # calculate deviations from true value and divide it by true value 
    # to make them comparable for each algorithm
    deviations = {algorithm: [] for algorithm in algorithms}

    for algorithm in algorithms:
        # if the result of any month is nan, skip the algorithm
        if np.isnan(predictions[algorithm]).any():
            continue
        for idx in range(2):
            deviations[algorithm].append(
                abs(predictions[algorithm][idx] - true_values[idx] /
                 true_values[idx]))

    # calculate average diviation for each algorithm
    deviations = {algorithm: np.average(values) for (algorithm, values) in deviations.items()}

    # select the algorithm with min deviation
    best_alg = min(deviations, key=deviations.get)

    # apply the best algorithm on data
    calculated_prediction = best_alg(df)

    with open('output/result.txt', 'w') as f:
        f.write(str(calculated_prediction))


def load_dataset_docker():
    """
    A function to load the dataset in the virtual environment
    """
    # load dataset                    
    with open('input/dataset.csv', 'r') as f:
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



def identify_worst_performers(df):
    """
    Find the 10% inaccurate performers in both months and identify the intersection of those.
    Adds 3 columns called 'exclude_10', 'exclude_10_1', 'exclude_10-2' to the dateframe.
    The first one is the intersection of the later ones. 1 means the participant was in
    10% low performance.
    For test, it uses the exclusions from the other month. For the actual prediction
    it uses the intersection of the two ('exclude_10')
    """
    # number of participants
    n_participants = len(df)
    n_to_exclude = int(n_participants * .1)
    # magnitude of the distance from the ture value
    distance_from_true_value_1 = abs(df['prediction_m1'] - df['true_value_m1'])
    distance_from_true_value_2 = abs(df['prediction_m2'] - df['true_value_m2'])

    # swap months, so it trains on one and tests on the other
    exclude_idx_2 = distance_from_true_value_1. \
                    sort_values(ascending=False)[:n_to_exclude].\
                    index.to_list()
    exclude_idx_1 = distance_from_true_value_2. \
                    sort_values(ascending=False)[:n_to_exclude].\
                    index.to_list()
    # find the intersection
    exclude_idx = list(set(exclude_idx_1) & set(exclude_idx_2))

    # make columns for exculstions; for now, make it all 0
    df[['exclude_10', 'exclude_10_1', 'exclude_10_2']] = 0

    # add excluxions to the column
    df.loc[exclude_idx, 'exclude_10'] = 1
    df.loc[exclude_idx_1, 'exclude_10_1'] = 1
    df.loc[exclude_idx_2, 'exclude_10_2'] = 1



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



def mean_remove_worst(df, month=None):
    """
    This function calculates the mean of the predictions after removing 
    bad performers in both months.
    """
    if month == None: # for the actual prediction
        prediction_var_name = 'prediction'
        exclusion_var_name = 'exclude_10'
    else: # if month was given
        # create variable names for the given month
        prediction_var_name = 'prediction_m' + str(month)
        exclusion_var_name = 'exclude_10_' + str(month)

    return df[df[exclusion_var_name] == 0][prediction_var_name].mean()



def median_remove_worst(df, month=None):
    """
    This function calculates the median of the predictions
    after removing worst performers in both months.
    """
    if month == None: # for the actual prediction
        prediction_var_name = 'prediction'
        exclusion_var_name = 'exclude_10'
    else: # if month was given
        # create variable names for the given month
        prediction_var_name = 'prediction_m' + str(month)
        exclusion_var_name = 'exclude_10_' + str(month)

    return df[df[exclusion_var_name] == 0][prediction_var_name].median()



def geo_mean(df, month=None):
    """
    This function calculates the geometric average of the predictions.
    To avoid overflowing memory, logarithm, mean and then exponantial functions are applied.
    """
    if month == None: # for the actual prediction
        prediction_var_name = 'prediction'
    else: # if month was given
        # create variable names for the given month
        prediction_var_name = 'prediction_m' + str(month)

    return np.exp(np.log(df[prediction_var_name]).mean())    



def diversify_responses(df, month=None):
    """
    Picks the responses with the highest diversity. 
    The idea came from a paper called 
    "Crowd Wisdom Relies on Agents Ability in Small Groups with a Voting Aggregation Rule"
    by Marc Keuschnigg and Christian Ganser.
    It assumes that predictions with highest diversity are most accurate.
    In n_samples, it goes through the predictions (without replacement) 
    and randomly picks one prediction,
    if the prediction increases the standard deviation of the prediction, it will
    be added. Otherwise, it picks another prediction.
    Then, it calculates the average for each sampling and average of the averages
    will be output value.

    """
    # set see for reproducability
    np.random.seed(2024)
    n_participants = len(df)
    if month == None: # for the actual prediction
        prediction_var_name = 'prediction'
    else: # if month was given
        # create variable names for the given month
        prediction_var_name = 'prediction_m' + str(month)
    
    n_samples = 1000
    predictions = []
    for trial in range(n_samples):
        all_ids = df.index.to_list()
        current_std = 1e-3
        first_id = np.random.choice(all_ids)
        all_ids.remove(first_id)
        prediction_list = [df[prediction_var_name][first_id]]
        id_list = [first_id]
        for _ in range(n_participants - 1):
            next_id = np.random.choice(all_ids)
            #consider each id only once
            all_ids.remove(next_id)
            next_prediction = df[prediction_var_name][next_id]
            next_std = np.std(prediction_list + [next_prediction])
            if current_std < next_std:
                prediction_list.append(next_prediction)
                id_list.append(next_id)
                current_std = next_std
        predictions.append(prediction_list)

    prediction_avgs = []
    for prediction_sample in predictions:
        prediction_avgs.append(np.mean(prediction_sample))

    return np.mean(prediction_avgs)



def top_k_mean(df, month=None):
    """
    This function finds the top predictors for each month. The idea came from a paper called
    "The wisdom of smaller, smarter crowds" by Goldstein et al.
    The k is also something to be found. Such that the function sorts the predictors based on their
    absolute distance from true value, then aggregates (average) their prediction from the first 5 up to
    the number of predictors. Then picks the closest average to the true value and the ids of those
    k best predictors. Then, the union of the ids from both months is used for final prediction.
    For each month prediction, it learns from one month and apply it to the other.
    """
    if month == None: # for the actual prediction
        prediction_var_name = 'prediction'
        ids_var_name = 'best_predictors'
    else: # if month was given
        # create variable names for the given month
        prediction_var_name = 'prediction_m' + str(month)
        ids_var_name = 'best_predictors_' + str(month)

    # since 5 is the min number of k, if predictors are less than 5, return NaN
    if len(df) < 5:
        return float('NaN')
    # keep ids in a dict
    ids = {}
    # swap months to learn from one and predict from the other
    ids['best_predictors_1'] = get_best_k(df, month=2)
    ids['best_predictors_2'] = get_best_k(df, month=1)

    # find the unision of best predictors in both months
    ids['best_predictors'] = list(set(ids['best_predictors_1']) | set(ids['best_predictors_2']))

    return df[df['id'].isin(ids[ids_var_name])][prediction_var_name].mean()



def get_best_k(df, month):
    """
    This function is part of the algorithm top_k_mean.
    The function sorts the predictors based on their absolute distance from true value, then aggregates (average) their prediction from the first 5 up to
    the number of predictors. Then picks the closest average to the true value and the ids of those
    k best predictors.
    """
    n_participants = len(df)
    # create variable names for the given month
    prediction_var_name = 'prediction_m' + str(month)
    true_v_var_name = 'true_value_m' + str(month)
    # sort df based on closeness to the true value
    results = df.iloc[(
        df[prediction_var_name] - df[true_v_var_name]
        ).abs().argsort()]
    
    distance_from_true_value = {}
    # find best value for k
    for k in range(5, n_participants + 1):
        distance_from_true_value[k] = \
            abs(results[prediction_var_name][:k].mean() - df[true_v_var_name][0])
    
    best_k = min(distance_from_true_value, key=distance_from_true_value.get)
    return results['id'][:best_k]



def CWM(df, month=None):
    """
    This function is based on Contribution Weighted Model
    (developed by DV Budescu, E Chen, 2014)
    This model wants to identify under-performing individuals and remove
    them from aggregation and weight others based on their contributions.
    First calculates the average of all predictors and find its distance from true value.
    Then calculates the average of everyone except the i th judge
    and caluclates its distance from true value. 
    Compares those two distances. If the average without the judge was closer the
    true value, the judge will be removed. Then, the judges with positive contribution
    remains and get normalized weight based on the amount of positive contribution
    (how much closer the average prediction gets after including the judge).

    For each month, the weights was learned from the other month. For the
    final prediction, the weights are averaged.
    """
    n_participants = len(df)
    if month == None: # for the actual prediction
        prediction_var_name = 'prediction'
        weights_var_name = 'best_predictors'
    else: # if month was given
        # create variable names for the given month
        prediction_var_name = 'prediction_m' + str(month)
        weights_var_name = 'best_predictors_' + str(month)
    
    # keep weights in a dict
    weights = {}
    # swap months to learn from one and predict from the other
    weights['best_predictors_1'] = get_weights(df, month=2)
    weights['best_predictors_2'] = get_weights(df, month=1)

    # to caclulate weights for the actual prediction we need to put 0 for 
    # people who did not show up in one of the months
    weights['best_predictors'] = weights['best_predictors_1'].add(
                                weights['best_predictors_2'],
                                fill_value=0) / 2

    # the mean() takes care of NaNs
    return (weights[weights_var_name] * df[prediction_var_name]).mean()



def get_weights(df, month):
    """
    This function is part of the algorithm CWM.
    Calculates the average of all predictors and find its distance from true value.
    Then calculates the average of everyone except the i th judge
    and caluclates its distance from true value. 
    Compares those two distances. If the average without the judge was closer the
    true value, the judge will be removed. Then, the judges with positive contribution
    remains and get normalized weight based on the amount of positive contribution
    (how much closer the average prediction gets after including the judge).
    """
    n_participants = len(df)
    
    # create variable names for the given month
    prediction_var_name = 'prediction_m' + str(month)
    true_v_var_name = 'true_value_m' + str(month)

    # use mean() for aggregation
    all_average = df[prediction_var_name].mean()

    #keep group's average without the i th individual
    one_out_averages = ((df[prediction_var_name].sum(0) - df[prediction_var_name]) / 
                        float(n_participants-1))

    # pick a merit function to check how far responses are from the true value
    # let's have the magnitude of the distance
    distance_from_true_value = abs(one_out_averages - df[true_v_var_name])
    # check the distance for the whole group
    gorup_distance = abs(all_average - df[true_v_var_name].to_list()[0])

    # calculated judge weights based on how they have changed distance from true value
    weight_judeges = gorup_distance - distance_from_true_value


    # find indices for positive wighted judges
    positive_weight_judeges = weight_judeges[weight_judeges > 0]

    # normalize weights
    normal_weights = positive_weight_judeges / positive_weight_judeges.sum()

    return normal_weights


# run the main function
if __name__ == "__main__":
    main()