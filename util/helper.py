import pandas as pd
import numpy as np

from scipy.stats import zscore, norm
from scipy.stats import skew
from scipy.stats import boxcox

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


class Filler:
        
    def fill_all(self, df, fill_method='0'):
        """Calls each method in Filler Class except of prediction methods"""
        
        df = self.fill_FloorNumber(df=df)
        print('FloorNumber filled.')
        df = self.fill_RenovationYear(df=df)
        print('RenovationYear filled.')
        df = self.fill_politics(df=df, method=fill_method)
        print('Politics filled.')
        df = self.fill_AreaProperty(df=df)
        print('AreaProperty filled.')
        df = self.fill_rest(df=df)
        print('Remaining Attributes filled.')
        
        return df
          
    def fill_FloorNumber(self, df):
        """
        Fills floor number with Median of each RealEstateTypeId and remainig with rounded mean of the Floor Number
        param df: Input Dataframe
        returns df: Output Dataframe
        """
        # Set values to nan which make no sense
        df.loc[df['FloorNumber'] >= 1000,'FloorNumber'] = np.nan
        df.loc[df['FloorNumber'] < 0,'FloorNumber'] = np.nan

        # Fill remaining values by rounded Mean of FloorNumbers
        fill_mean = round(np.mean(df['FloorNumber']), 0)
        df['FloorNumber'] = df['FloorNumber'].fillna(value=fill_mean)
        
        return df
    
    def fill_AreaProperty(self, df):
        """ THis function is only applicable on test set because 0 values are treated as NA.
        We fill it with 0
        """
        df['AreaProperty'] = df['AreaProperty'].fillna(value=0)
        
        return df
    
    def fill_RenovationYear(self, df):
        """Fills RenovationYear with value 0"""
        fill_value = 0
        df['Renovationyear'] = df['Renovationyear'].fillna(value=fill_value)
        
        return df
        
    def fill_politics(self, df, method):
        """Fills all politics columns with given method as parameter"""
        politics = [i for i in df.columns if 'politics' in i]
        
        if method == '0':
            df[politics] = df[politics].fillna(value=0)
        if method == 'median':
            for col in politics:
                df[col] = df[col].fillna(value=np.median(df[col].dropna()))
        elif method == 'mean':
            for col in politics:
                df[col] = df[col].fillna(value=np.mean(df[col].dropna()))
            
        return df
    
    def fill_rest(self, df):
        """Fills remaining values with mean value of each column"""
        is_null = df.isnull().sum(axis=0)[df.isnull().sum(axis=0) > 0].index.to_list()
        values = df[is_null].mean(skipna=True)
        df.loc[is_null] = df.loc[is_null].fillna(value=values)
        
        return df

    
class NewFeatures:
        
    def add(self, df):
        # Add total square meters
        df['AreaTotal'] = df['AreaLiving'] + df['AreaProperty']
        # add has additional space attribute
        df['has_space'] = np.where(df['AreaProperty'] > 0, 1, 0)

        # difference between renovationyear and built year
        df['Renovation-Built'] = df['Renovationyear'] - df['BuiltYear']
        
        # Add column is renovated
        df['is_renovated'] = np.where(df['Renovation-Built'] > 0, 1, 0)

        # Inserts per zip
        new_attr = df.groupby('Zip').count().drop_duplicates().iloc[:,0].to_dict()
        df['inserts_per_zip'] = df['Zip'].replace(to_replace=new_attr.keys(), value=new_attr.values()) 

        return df
    
    def combine_cat_features(self, df):
        object_cols = df.select_dtypes(include='object')
        # Add new column with GroupNameDe and Name where the data comes from
        df['GroupNameDe' + 'Name'] = df['GroupNameDe'] + ', ' + df['Name']
        # Add new feature of object and stateshort
        df['GroupNameDe' + 'StateShort'] = df['GroupNameDe'] + ', ' + df['StateShort'] 

        return df

       
class Encoder:
    
    def __init__(self):
        self.categorical_encoding_dict = None
        self.target_encoding_dict = None
        self.count_encodings = None
        
    def count_encoding(self, df):
        """Encodes categorical Features with their number of occurencies in the dataframe."""
        categorical_features = df.select_dtypes(include='object').columns.to_list()
        count_encodings = {}

        for col in categorical_features:
            count_encodings[col] = ce.CountEncoder()
            df[col] = count_encodings[col].fit_transform(df[col])

        self.count_encodings = count_encodings

        return df
        
    def encode(self, df, encoding='label'):
        """Function which uses one of class methods
        
        encoding parameters: 
            - 'label' for label-encoding
            - 'binary' for pandas dummy encoding
            - 'target' for target encoding
            
        returns: encoded Dataframe
        """
        if encoding == 'label':
            df = self.categorical_encoding(df)
        elif encoding == 'binary':
            df = self.binary_encoding(df)
        elif encoding == 'target':
            df = self.target_encoding(df)
         
        return df
    
    def categorical_encoding(self, df):
        """Encodes the object with categorical code"""        
        df_cat = df.select_dtypes(include='object')
        print('columns with categorical values: ', df_cat.columns)
    
        try:
            for i in ['StreetAndNr']: # Ensure to not encode street and nr
                df_cat = df_cat.drop(i, axis=1)
        except:
            pass
       
        encodings = {}
        for col in df_cat.columns:
            encodings[col] = dict(zip(df_cat[col].unique(), [i for i in range(len(df_cat[col].unique()))]))
        
        # Add encodings to class instance  
        self.categorical_encoding_dict = encodings
        
        # Encode categories 
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].replace(encodings[col])
            
        # Set categories in in df to encoded categories
        for col in df_cat.columns:
            df[col] = df_cat[col]
        
        return df
    
    def binary_encoding(self, df):
        """Encodes the object with binary code"""
        
        df = df.copy()
        
        categories = df.select_dtypes(include='object').columns.to_list()
        categories.remove('LastUpdate')
        categories.remove('Locality')
        
        for cat in categories:
            df_encoded = pd.get_dummies(df[cat])
            df = df.drop(cat, axis=1)
            df = pd.concat([df, df_encoded], axis=1)
            
        return df
    
    def target_encoding(self, df):
        """
        This function encodes all categorical attributes with target encoding. Encoded values are probability of value in dataframe
    
        param df: pandas dataframe
        param column: column of dataframe
        returns: df with encoded column
        """
        categories = list(df.select_dtypes(include='object').columns)
        
        print('Categories to Encode: ', categories)
 
        if 'LastUpdate' in categories:
            categories.remove('LastUpdate')
        if 'Locality' in categories:
            categories.remove('Locality')
        if 'StreetAndNr' in categories:
            categories.remove('StreetAndNr')
        
        # Get all relative counts per category
        for cat in categories:
            cat_counts = df.groupby(cat).count().iloc[:,0].reset_index()
            cat_counts.columns = [cat,'counts']
            summed_up = cat_counts['counts'].sum(axis=0)
            cat_counts['rel'] = cat_counts['counts'] / summed_up
    
            # Create dict and replace
            replace_dict = cat_counts.drop(['counts'], axis=1).set_index(cat).to_dict()['rel']
            df[cat] = df[cat].replace(replace_dict)
        
        # Save dictionary to object 
        self.target_encoding_dict = replace_dict
    
        return df

    
class Dropper:
        
    def drop_all(self, df):
        """Calls each method in Dropper Class"""
        df = self.drop_col(df=df)
        df = self.drop_low_missing_values(df=df)
        
        return df
             
    def drop_col(self, df):
        """Drops columns with redundant information"""
        columns_2_drop = ['NoisePollutionRailwayL', 'NoisePollutionRailwayS',
                            'NoisePollutionRoadS', 'NoisePollutionRoadL',
                            'PopulationDensityL', 'PopulationDensityS',
                            'RiversAndLakesL', 'RiversAndLakesS',
                            'WorkplaceDensityS', 'WorkplaceDensityL',
                            'ForestDensityS', 'ForestDensityL', 'StreetAndNr'] 
                
        for col in columns_2_drop:
            try:
                df = df.drop(col, axis=1)
            except:
                print('Not found {} in axis. Already removed'.format(col))
            
        return df       
    
    def drop_low_missing_values(self, df):
        is_null = df.isnull().sum()
        columns_low_outlier = is_null[(is_null > 0) & (is_null < 100)].index.to_list()

        # Drop all rows where the outliers are only 2 
        for col in columns_low_outlier:
            df = df.drop(df[df[col].isnull()].index, axis=0)

        return df
