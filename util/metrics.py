import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy.stats import zscore


class Metrics:
    """
    This class is the for the purpose to analyze our models.
    
    Init Parameters:
    - y_pred = predicted values by model (Numpy-Array shape of (-1,1))
    - y_test = acutal values of the data (Numpy-Array shape of (-1,1))
    """
    
    def __init__(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
        self.residuals = None
        self._residuals()
        
        
    def _residuals(self):
        """Estimates the residuals of given y_pred and y_test"""
        if len(self.y_pred.shape) < 2:
            self.y_pred = np.expand_dims(self.y_pred, 1)
        elif len(self.y_test.shape) < 2:
            self.y_test = np.expand_dims(self.y_test, 1)
            
        #print('Shape of Y_pred: {}, Shape of Y_test: {}'.format(self.y_pred.shape, self.y_test.shape))
        
        if (self.y_pred.shape[1] == 1) & (self.y_test.shape[1] == 1):
            self.residuals = self.y_test - self.y_pred 
        else:
            print("Shapes of inputs: {}, {}\n Type of input don't match operation.".format(self.y_pred.shape,
                                                                                                        self.y_test.shape))
              
    def get_summary(self):
        """Calls all function within class"""
        print('Mean ABS Error: ', self.mean_abs_error(), 
              '\nMean ABS Percentage Error: ', self.mean_abs_percentage_error(), 
              '\nMedian ABS Error: ', self.median_abs_error())
        p1 = self.plot_residuals()
        p2 = self.plot_qq()

        return p1, p2
        
            
    def median_abs_error(self):
        """
        Calculates median absolute error of the Residuals
        
        f = median(abs(y_test - y_pred))
        
        """
        return round(np.median(abs(self.residuals)),2)
    
    def mean_abs_error(self):
        """Calculates Mean Absolute Error of the residuals
        
        f = (1 / n) * sum(abs(y_test - y_pred))
        
        returns: MAE
        """
        return round(np.mean(abs(self.residuals)),2)
    
    def mean_abs_percentage_error(self):
        """Calculates mean absolute percentage error:
        
        f = (1/n) * sum(abs(y_test - y_pred)/ abs(y_test))
        
        returns: MAPE
        """
        return np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) 
    
    ################## Plotting #######################
        
    def plot_residuals(self):
        """
        This function calculates the redisuals on the object and returns a plot which shows the distrbution of 
        the residuals and the scatter plot.
        """           
        residuals = self.residuals.reshape(1,-1)[0].tolist()
        residuals = zscore(residuals) # standardize Residuals
        
        tmp = pd.DataFrame(data={'y_test':self.y_test.reshape(1,-1)[0].tolist(), 
                                 'y_pred':self.y_pred.reshape(1,-1)[0].tolist(), 
                                 'residuals':residuals})
        tmp = tmp.sort_values(by='y_test')
        
        fig, ax = plt.subplots(1,2,figsize=(14,5))
        p1 = sns.scatterplot(x=tmp['y_pred'], y=tmp['residuals'], ax=ax[0], alpha=0.1, linewidth=0)
        p2 = sns.distplot(tmp['residuals'], kde=False, ax=ax[1])
        p1.hlines(y=0, xmin=min(tmp['y_pred']), xmax=max(tmp['y_pred']), linestyles='--')
        p1.set_title('Tukey-Anscombe Diagram of Residuals')
        p1.ticklabel_format(style='plain')
        p1.set_ylabel('Standardized Residuals')
        p2.set_title('Histogram of Residuals')
        p2.set_ylabel('Count Residuals')
        p2.set_xlabel('Standardized Residuals')
        p2.ticklabel_format(style='plain')
        plt.close()
        
        return fig
        
        
    def plot_qq(self):
        """
        Plots residuals against sample quantiles of normal distribution with respective value. Calls function scipy.stats.probplot
        returns: QQ-Plot
        """
        # Generate sample quantiles and theoretical qauntiles of norm distribition 
        sample_quantiles = np.quantile(self.residuals.reshape(1,-1)[0], q=np.arange(0,1, 0.01)) 
        sample_mean, sample_std = norm.fit(self.residuals.reshape(1,-1)[0])
        theoretical_quantiles = norm.ppf(q=np.arange(0,1,0.01), loc=sample_mean, scale=sample_std) 

        quantiles = np.vstack((theoretical_quantiles, sample_quantiles))
        
        # Mask oout unlogical values
        mask = (np.isinf(quantiles)) | (np.isnan(quantiles))
        mask = mask.any(axis=0) == False
        quantiles = quantiles[:, mask]

        # Plot
        fig, ax = plt.subplots(1,1)
        p = sns.regplot(quantiles[0], quantiles[1], ci=False, line_kws={'color':'red'})
        p.set_title('QQ-Plot')
        p.set_xlabel('Theoretical Quantiles (Normal Distribution)')
        p.set_ylabel('Sample Quantiles')
        plt.close()
        
        return fig
        