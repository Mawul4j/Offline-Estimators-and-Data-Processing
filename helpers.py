# includes helper functions for SAMMI data analysis


import time
import datetime
import dateutil.parser
import pandas as pd
from time import mktime as mktime

def convertToLong(x):
	
    """
    Takes a columns of dates in string format
    Returns: unix time
    """
    x = dateutil.parser.parse(x).strftime("%Y-%m-%d %H:%M:%S")
    val  = int(mktime(datetime.datetime.strptime(x,  '%Y-%m-%d %H:%M:%S').timetuple()))
    
    return val



def file_combiner(path, ext):
    """
    this function combines csv files from a directory
    path : directory (str)
    ext: file extensions (str)
    """
    filenames = []
    for fil in os.listdir(path):
        if fil.endswith(ext):
            names = str(os.path.join(path, fil))
            filenames.append(names)
            
    combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )
    return combined_csv


def day_extractor(series):
    """take series -> outputs series
    returns the extracted days from date format dd/mm/year """
    
    days = list()
    #days.append(1)
    series = list(series)
    value = 1
    #prev_val =list(series)[0]
    #next_val = list(series)[1]
    days.append(1)
    for i in range(1, len(series)):  
        prev_val = series[i-1].split('/')[0]
        next_val = series[i].split('/')[0]
            
        if prev_val == next_val:
            days.append(value)
            prev_val = series[i-1].split('/')[0]
            next_val = series[i].split('/')[0]

        else:
            value += 1
            days.append(value)
            prev_val = series[i-1].split('/')[0]
            next_val = series[i].split('/')[0]
    
    assert(len(days) == len(series))  
    return days


def remove_outlier(df_in, col_name):
    '''
    takes a data frame and column,
    cap off the values in the column with the 5 and 95%tile
    return the data frame with the updated column
    '''
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    
    return df_out



def changed(a, b):
    '''
    Predicate: checking if b < a
    '''
    return True if (b < a) else False

def pedo_finder(L):
    '''
    list --> scalar
    takes list of increasing values
    returns number of steps
    '''
    counter = 0
    for i in range(len(L)-1):
        prev = L[i]
        curr =  L[i+1]

        if changed(prev, curr):
            counter += prev
        else:
            continue
    return (counter + L[-1])

   
def splitter_fx(x):
    '''
    takes values and return split by ',' or string
    '''
    if pd.isnull(x):
        return 
    else:
        #print(i.split(','))
        
        if len(x.split(',')) == 1:
            return x.split(',')[0]
        else:
            return 'two_or_more'


def labels_creator(cache_list, cache_dict, col):
    """
    Takes in a a dictionary of strategies and
    split them into binary variables.
    -----------------------------------------------------------
    cache_dict -- dictionary of strategies each with value = 0
    cache_list -- list of dictionaries to be converted into dataframe
    col -- column to be counted
    
    """
    for val in col:
        #resets all values to 0
        cache_dict = {key: 0 for key in cache_dict}
        
        if type(val) == float: 

            print('this is a nan!')
            cache_dict = {key: val for key in cache_dict}
            cache_list.append(cache_dict)
            continue
        else:
            val_split = val.split(',')
            for strat in val_split:
                if strat in cache_dict:
                    cache_dict[strat] += 1
                else:
                    continue
            cache_list.append(cache_dict)
            
    return cache_list





