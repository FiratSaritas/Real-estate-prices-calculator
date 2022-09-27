import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np


class Util:
    """
    This class is a helper class which does import the dataframe and appleis some given operations on it to decrease the memory
    size of notebook
    
    param path: relative path of data
    param keep_columns: Specify columns to import 
    param filter_rooms: threshold to filter rooms
    
    returns: preprocessed pandas dataframe
    """
    
    def __init__(self, path, is_test=False):
        self.path = path
        self.df = None
        self._read_df()
        self._set_datatypes()
        
        if is_test:
            self._fill_AreaProperty()
                
    def _read_df(self):
        self.df = pd.read_csv(self.path) 
    
    def _set_datatypes(self):
        """
        Sets appropriate data types to the data.
        
        :params: self
        :returns: df
        """
        self.df['AreaLiving'] = self.df['AreaLiving'].astype('int32')
        self.df['AreaProperty'] = self.df['AreaProperty'].astype('int32', errors='ignore') # Ignore error due NA in Areaproperty in testset
        self.df['BuiltYear'] = self.df['BuiltYear'].astype('int32')
        self.df['HouseObject'] = self.df['HouseObject'].astype('int8') # only 1, 0
        self.df['Rooms'] = self.df['Rooms'].astype('int32')
        self.df['Zip'] = self.df['Zip'].astype('int32')
        self.df['location_has_street'] = self.df['location_has_street'].astype('int8') # only 1, 0
        self.df['location_is_complete'] = self.df['location_is_complete'].astype('int8') # only 1, 0
        
    def _fill_AreaProperty(self):
        """
        fills AreaProperty missing values.
        """
        self.df['AreaProperty'] = self.df['AreaProperty'].fillna(value=0)
        
    def return_dataframe(self):
        return self.df
            
        
    
    
    
    