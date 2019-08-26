import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dateutil.parser import parse


def feature_extract(df, target='grade'):
    loe_encoding = {np.nan: -1, 'Less than Secondary': 0, 'Secondary': 1, 
                    "Bachelor's": 2, "Master's": 3, 'Doctorate': 4}

    baseline = parse('2012-01-01')

    def yob_ranges(year):
        base = 1940
        index = -1
        while year > base:
            index += 1
            base += 10
        return index

    # Removendo 'certified' para evitar viez
    feat_df = pd.DataFrame(df[['viewed', 'explored',  'LoE_DI', 'YoB', 'final_cc_cname_DI',
                                      'gender', 'nevents', 'ndays_act', 'nplay_video', 'start_time_DI', 'last_event_DI',
                                      'nchapters', 'nforum_posts', 'incomplete_flag', target]])

    feat_df['LoE_DI'] = feat_df['LoE_DI'].map(loe_encoding)
    feat_df['YoB'] = feat_df['YoB'].map(yob_ranges)
    feat_df['gender_f'] = feat_df['gender'].map(lambda g: 1 if g == 'f' else 0)
    feat_df['gender_m'] = feat_df['gender'].map(lambda g: 1 if g == 'm' else 0)
    feat_df['nevents'].replace(np.nan, -1, inplace=True)
    feat_df['ndays_act'].replace(np.nan, -1, inplace=True)
    feat_df['nplay_video'].replace(np.nan, -1, inplace=True)
    feat_df['nchapters'].replace(np.nan, -1, inplace=True)
    feat_df['nforum_posts'].replace(np.nan, -1, inplace=True)
    feat_df['incomplete_flag'].replace(np.nan, -1, inplace=True)
    if target == 'grade':
        feat_df['grade'].replace(np.nan, -1, inplace=True)
    elif target == 'certified':
        feat_df['certified'].replace(np.nan, 0, inplace=True)

    feat_df['cc_eua'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in ['United States'] else 0)
    feat_df['cc_central_north_america'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in ['Mexico', 'Canada', 'Other North & Central Amer., Caribbean'] else 0)
    feat_df['cc_south_america'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in ['Colombia', 'Brazil', 'Other South America'] else 0)
    feat_df['cc_europe'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in
        ['France', 'Russian Federation', 'Other Europe', 'United Kingdom', 'Ukraine', 'Spain', 'Greece', 'Germany',
         'Poland', 'Portugal'] else 0)
    feat_df['cc_asia'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in
        ['India', 'Other South Asia', 'Japan', 'Other Middle East/Central Asia', 'Other East Asia', 'Bangladesh',
         'China', 'Pakistan', 'Philippines'] else 0)
    feat_df['cc_oceania'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in ['Australia', 'Other Oceania', 'Indonesia'] else 0)
    feat_df['cc_africa'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in ['Other Africa', 'Nigeria', 'Egypt', 'Morocco'] else 0)
    feat_df['cc_unknown'] = feat_df['final_cc_cname_DI'].map(lambda x: 1 if x in ['Unknown/Other'] else 0)

    feat_df['start_time_DI'] = feat_df['start_time_DI'].map(lambda d: (parse(str(d)) - baseline).days)
    feat_df['last_event_DI'].replace(np.nan, '2011-12-31', inplace=True)
    feat_df['last_event_DI'] = feat_df['last_event_DI'].map(lambda d: (parse(str(d)) - baseline).days)
    feat_df['duration'] = feat_df['last_event_DI'] - feat_df['start_time_DI']
    feat_df['duration'] = feat_df['duration'].map(lambda d: d if d > 0 else 0)
    feat_df = feat_df.drop(columns=['gender', 'final_cc_cname_DI', 'start_time_DI', 'last_event_DI'])


    return feat_df


def normalize(df):
    return pd.DataFrame(preprocessing.normalize(df, norm='l2'))


def split_xy(df, label='grade'):
    return df.loc[:, df.columns != label], pd.DataFrame(df[label])


def train_test_val_split(X, y, test_size=0.25, val_size=0.25, random_state=42):
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, [],  y_train, y_test, []

def example():
    return 'aaaa'