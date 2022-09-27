import pandas as pd
import numpy as np

from scipy.stats import zscore, norm
from scipy.stats import skew
from scipy.stats import boxcox

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


class Tidy:
    """
    This class is used as function library for any kinds tidying the dataframe.
    
    Usage:
    >>> tidy = Tidy()
    >>> df = tidy.remove_outliers_by_zscore(df=df)
    """

    def remove_outliers_purchase_price_z(self, df):
        """
        Removes Outliers for predictor variable for Regression Model (PurchasePrice)
        
        param df: pandas dataframe
        returns df: df with removed outliers of PurchasePrice
        """
        scores = zscore(np.log(df['PurchasePrice']))
        drop_idx = (scores >= 3) | (scores <=-3)
        
        return df[~drop_idx]
    
    def remove_outliers_by_box(self, df, column):
        """Removes Outliers with by Q1 and Q3 +- 1.5 *IQR """
        upperq, lowerq = df[column].quantile(0.75) , df[column].quantile(0.25)
        iqr = upperq - lowerq
        iqr = 1.5 * iqr

        return df[(df[column] < upperq + iqr) & (df[column] > lowerq - iqr)]
    
    def remove_outliers_by_zscore(self, df, column, cut_off):
        """Removes outliers acc. zscore cut_off value. Input column should be normal distributed.
        
        param df: pandas dataframe
        param column: column to remove outliers from
        param: cut_off: Zscore to cut on both tailes
        returns df: df with removed outliers
        """
        shape_before = len(df)
        scores = zscore(df[column])
        drop_idx = (scores >= cut_off) | (scores <= -cut_off)
        
        print('Count Removed Outliers: {}'.format(shape_before - len(drop_idx)))
        
        return df[~drop_idx]
        
     
    def drop_column(df):
        drop_list= ['StreetAndNr','LastUpdate','HouseObject', 'Id', 'location_has_street', 'location_is_complete',
                    'Name','StateShort', 'GroupNameDe', 'Locality', 'FloorNumber',
                    'gde_politics_bdp','gde_politics_cvp','gde_politics_evp','gde_politics_fdp',                'gde_politics_glp','gde_politics_gps','gde_politics_pda','gde_politics_rights','gde_politics_sp','gde_politics_svp']
        for i in drop_list: 
            if i in df:
                df.pop(i)
            else: pass 
        return df
      
    def filter_Renovationyears(df):
        frames = [df[(df['BuiltYear'] < df['Renovationyear'])], df[df['Renovationyear'].isnull() == True]]
        filt = pd.concat(frames)
        return filt
    
    def filter_Rooms(df):
        filt = df[df['Rooms'] < 25]
        return filt
    
    def filter_floor(df):
        filt = df[df['FloorNumber'] < 31]
        filt = filt[filt['FloorNumber'] > -1]
        return filt
    
    
class Filler:
        
    def fill_all(self, df, predict=False, fill_method='median'):
        """Calls each method in Filler Class except of prediction methods"""
        if predict:
            df = self.fill_RenovationYear(df=df)
            df = self.fill_AreaProperty(df=df)
            
            cols = [i for i in df.columns if 'politics' in i]
            cols.append('FloorNumber')
            
            for col in cols:
                df = self.predict_na_continous(df, col)
            df = self.fill_rest(df=df)
        else:
            df = self.fill_FloorNumber(df=df)
            df = self.fill_RenovationYear(df=df)
            df = self.fill_politics(df=df, method=fill_method)
            df = self.fill_AreaProperty(df=df)
            df = self.fill_rest(df=df)
        
        return df
          
    def fill_FloorNumber(self, df):
        """
        Fills floor number with Median of each RealEstateTypeId and remainig with rounded mean of the Floor Number
        param df: Input Dataframe
        returns df: Output Dataframe
        """
        # RealEstateTypeId grouped Median
        grouped_values = df.groupby(by='RealEstateTypeId').median()['FloorNumber']
        grouped_values = grouped_values.dropna(axis=0)
        
        # Replace all NA-Values by grouped belonging grouped Median of RealEstateTypeId
        for i in grouped_values.index:
            tmp = df.loc[df['RealEstateTypeId'] == i,:].copy()
            tmp.loc[:,'FloorNumber'] = tmp.loc[:,'FloorNumber'].fillna(value=grouped_values.loc[i])
            df.loc[df['RealEstateTypeId'] == i, :] = tmp

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
        
    def fill_politics(self, df, method='median'):
        """Fills all politics columns with given method as parameter"""
        politics = [i for i in df.columns if 'politics' in i]
        
        if method == 'median':
            for col in politics:
                df[col] = df[col].fillna(value=np.median(df[col].dropna()))
        elif method == 'mean':
            for col in politics:
                df[col] = df[col].fillna(value=np.mean(df[col].dropna()))
            
        return df
    
    def fill_rest(self, df):
        """Fills remaining values"""
        return df.fillna(value=df.mean(axis=0))
    
    
    def predict_na_discrete(self, df, column):
        """Predicts outliers of a given categorical attribute with a decision tree classifier
    
        df: pandas dataframe
        column: column to predict outliers
        returns: df with predicted outliers for given input column
        """
        # Check if given attribute is numeric and if then encode it
        if hasattr(df[column], 'str'):
            encoder = LabelEncoder()
            df.loc[:, column] = encoder.fit_transform(df.loc[:, column])
            is_obj = True
        else:
            is_obj = False
    
        ## Prepare data to predict missing values, select only columns with no missing values
        columns_with_na = df.isna().sum(axis=0)
        columns_with_na = columns_with_na[columns_with_na > 0].index.to_list()
    
        #Split Dataset and get all indexes with NA in FloorNumber
        idx_number = np.array(df[df[column].isna() == False].index)
        idx_NA = np.array(df[df[column].isna() == True].index)
    
        X, y = df.loc[idx_number], df.loc[idx_number, column] 
    
        # Drop all columns in X with NA
        for i in columns_with_na:
            if i in X.columns.to_list():
                X = X.drop(i, axis=1)
            else:
                pass
        # Select Only dtypes without objects    
        X = X.select_dtypes(exclude='object')

        #print('Missing Values in the Datframe?', X.isna().sum().sort_values().any()) # Check if there are NA_Values left

        # Fit Model
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        print('Accuracy: ', tree.score(X, y))

        # Prepare Data for Prediction of NA
        predictors = df.loc[idx_NA].drop(columns_with_na, axis=1).select_dtypes(exclude='object')

        # Make Prediction for NA-Indices
        fill_values = tree.predict(predictors)

        # Fill in Values in dataframe
        df.loc[idx_NA, column] = fill_values
    
        # Reverse Encoding if categegorical variable as string was given
        if is_obj:
            df.loc[:, column] = encoder.inverse_transform(df.loc[:, column])
    
        return df
    
    def predict_na_continous(self, df, column):
        """Predicts outliers of a given continous attribute with a decision tree classifier
    
        df: pandas dataframe
        column: column to predict outliers
        returns: df with predicted outliers for given input column
        """
    
        ## Prepare data to predict missing values, select only columns with no missing values
        columns_with_na = df.isna().sum(axis=0)
        columns_with_na = columns_with_na[columns_with_na > 0].index.to_list()
    
        #Split Dataset and get all indexes with NA in FloorNumber
        idx_number = np.array(df[df[column].isna() == False].index)
        idx_NA = np.array(df[df[column].isna() == True].index)
    
        X, y = df.loc[idx_number], df.loc[idx_number, column] 
    
        # Drop all columns in X with NA for Tree Model
        for i in columns_with_na:
            if i in X.columns.to_list():
                X = X.drop(i, axis=1)
            else:
                pass
        # Select Only dtypes without objects    
        X = X.select_dtypes(exclude='object')

        #print('Missing Values in the Datframe?', X.isna().sum().sort_values().any()) # Check if there are NA_Values left

        # Fit Model
        tree = DecisionTreeRegressor()
        tree.fit(X, y)
        print('R2-Score: ', tree.score(X, y))

        # Prepare Data for Prediction of NA
        predictors = df.loc[idx_NA].drop(columns_with_na, axis=1).select_dtypes(exclude='object')

        # Make Prediction for NA-Indices
        fill_values = tree.predict(predictors)

        # Fill in Values in dataframe
        df.loc[idx_NA, column] = fill_values
    
        return df
    

class Transformer:
    
    def handle_skewness(self, df):
        """Function corrects skewness with boxcox transformation"""
        # Calculate skewness per attribute
        skews = df.skew()
        skews = skews[(skews > 0.5) | (skews < -0.5)]
        skews = list(skews.index)
    
        # Remove by eye checked attribtues which makes no sense to transform
        remove = ['RealEstateTypeId', 'is_renovated', 'location_has_street', 'Name']
        for i in remove:
            if i in skews:
                skews.remove(i)
            else:
                pass
        print('Transform Attributes: ', skews)
        # Transform all values in skews list
        for i in skews:
            df.loc[:, i], _ = boxcox(x=df.loc[:, i] + 1, lmbda=None)
    
        return df 
    
class Encoder:
    
    def __init(self):
        self.categorical_encoding_dict = None
        self.target_encoding_dict = None
        
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
