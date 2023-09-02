# all_stats
 Python library for automated statisticals tests for beginners

# Installation 

1. You need a python 3.8's version or greater to run all dependancies correcly
2. Clone the latest version of the library and requirements.txt with `git clone [url] -b latest`
3. Dependencies can be installed with the command line : `pip3 install -r requirements.txt` (python virtual environment is highly recommended)  
4. Move **correlation_test.py** into your python work directory (or python virtual environment directory)<br>  
 
# Usage

Once the set up completed, import the correlation test module as a library in a Python script `import correlation_test as ct`  
Current version cover all automated statistical test as describing below :   
* Pearson Chi-2's test (Categorical correlation) and V Cramer (correlation force measurement)
* Spearman's test (quantitative correlation)
* Shapiro's test (Normality test)  <br>

**Upcoming test** : 
* Anova's test 
* Pearson's test (quantitative)
* Student test

**3. Example with Pearson Chi-2's test** 

