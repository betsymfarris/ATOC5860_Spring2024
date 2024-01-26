#!/usr/bin/env python
# coding: utf-8

# #  Application Lab #1 ATOC5860 Objective Analysis - bootstrapping
# ##### Originally coded by Prof. Kay (CU) with input from Vineel Yettella (CU ATOC Ph.D. 2018)
# ##### last updated January 16, 2024
# 
# ### LEARNING GOALS:
# 1) Use an ipython notebook to read in csv file, print variables, calculate basic statistics, do a bootstrap, make histogram plot
# 2) Hypothesis testing and statistical significance testing using bootstrapping
# 3) Contrast results obtained using bootstrapping with results obtained using a t-test
# 
# ### DATA and UNDERLYING SCIENCE MOTIVATION:  
# In this notebook, you will analyze the relationship between Tropical Pacific Sea Surface Temperature (SST) anomalies and Colorado snowpack. Specifically, you will test the hypothesis that December Pacific SST anomalies driven by the El Nino Southern Oscillation affect the total wintertime snow accumulation at a mountain pass in Colorado.  When SSTs in the central Pacific are anomalously warm/cold, jet and precipitation locations can change. But do these atmospheric teleconnections affect total Colorado snow accumulation in the following winter? This notebook will guide you through an analysis to investigate the connections between December Nino3.4 SST anomalies (in units of °C) and the following April 1 Berthoud Pass, Colorado Snow Water Equivalence (in units of inches). Note that SWE is a measure of the amount of water contained in the snowpack.  To convert to snow depth, you multiply by ~5 (the exact value depends on the snow density).
# 
# The data have already been munged into a file called 'snow_enso_data_1936-2022.csv'. The Berthoud Pass SWE data are from: https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/ and
# https://wcc.sc.egov.usda.gov/nwcc/rgrpt?report=snowmonth_hist&state=CO. The Nino3.4 data are from: https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino34/

# ### First, let's load packages, read in data, look at your data

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# this enables plotting within notebook

import matplotlib   # library for plotting
import matplotlib.pyplot as plt #  later you will type plt.$COMMAND
import numpy as np   # basic math library  you will type np.$STUFF  e.g., np.cos(1)
import pandas as pd  # library for data analysis for text files (everything but netcdf files)
import scipy.stats as stats # imports stats functions https://docs.scipy.org/doc/scipy/reference/stats.html 


# In[5]:


### Read in the data
filename='snow_enso_data_1936-2022.csv'
data=pd.read_csv(filename,sep=',')
data.head()


# In[6]:


### Print the data column names
print(data.columns[0])
print(data.columns[1])
print(data.columns[2])


# In[9]:


### Print the data values - LOOK AT YOUR DATA.  
### check out what happens when you remove .values ??
print(data['Year'].values)
print(data['BerthoudPass_April1SWE_inches'].values)
print(data['Nino34_anomaly_prevDec'].values)


# ### Question 1: Composite Loveland Pass, Colorado snowpack data.
# 
# In other words - Find April 1 SWE in all years, in El Nino years (conditioned on Nino3.4 being 1 degree C warmer than average), and in La Nina years (condition on Nino3.4 being 1 degree C cooler than average). 
# 
# Make a table showing the results.

# In[5]:


### Calculate the average snowfall on April 1 at Berthoud Pass, Colorado
SWE_avg=data['BerthoudPass_April1SWE_inches'].mean()
SWE_std=data['BerthoudPass_April1SWE_inches'].std()
N_SWE=len(data.BerthoudPass_April1SWE_inches)
print(f'Average SWE (inches): {np.round(SWE_avg,2)}')
print(f'Standard Deviation SWE (inches): {np.round(SWE_std,2)}')
print(f'N: {np.round(N_SWE,2)}')


# In[20]:


# not sure if this is what the Question 1 is asking
SWE_all = data['BerthoudPass_April1SWE_inches']
SWE_ElNino = data[data.Nino34_anomaly_prevDec>1.0]
SWE_LaNina = data[data.Nino34_anomaly_prevDec<1.0]
SWE_ElNino.head()


# In[19]:


SWE_LaNina.head()


# In[25]:


### Print to figure out how to condition and make sure it is working.  Check out if new to Python.
# print(data.Nino34_anomaly_prevDec>1) ## this gives True/False
# print(data[data.Nino34_anomaly_prevDec>1])  ## where it is True, values will print



### Calculate the average SWE when it was an el nino year
SWE_avg_nino=data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'].mean()
SWE_std_nino=data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'].std()
N_SWE_nino=len(data[data.Nino34_anomaly_prevDec>1.0].BerthoudPass_April1SWE_inches)
print(f'Average SWE El Nino (inches): {np.round(SWE_avg_nino,2)}')
print(f'Standard Deviation SWE El Nino (inches): {np.round(SWE_std_nino,2)}')
print(f'N El Nino: {np.round(N_SWE_nino,2)}')


# In[7]:


### Calculate the average SWE when it was an la nina year
SWE_avg_nina=data[data.Nino34_anomaly_prevDec<-1.0]['BerthoudPass_April1SWE_inches'].mean()
SWE_std_nina=data[data.Nino34_anomaly_prevDec<-1.0]['BerthoudPass_April1SWE_inches'].std()
N_SWE_nina=len(data[data.Nino34_anomaly_prevDec<-1.0].BerthoudPass_April1SWE_inches)
print(f'Average SWE La Nina (inches): {np.round(SWE_avg_nina,2)}')
print(f'Standard Deviation SWE La Nina (inches): {np.round(SWE_std_nina,2)}')
print(f'N La Nina: {np.round(N_SWE_nina,2)}')


# ### Question 2: Use hypothesis testing to assess if the differences in snowpack are statistically significant. Write your hypothesis and the 5 steps you plan to apply here.
# 
# #### (insert your text here)
# Is the average SWE of Berthoud Pass consistent with the null hypothesis that the El Nino Southern Oscillation (December Pacific SST anomalies) does not change total wintertime snow accumulation?
# 
# 1) used alpha = 0.05, 95% confidence interval
# 2) H_0: mean_SWE_anomalous_year = average snowpack for all years,
# 3) 
# 4) 
# 

# ### Question 3: Let's bootstrap to evaluate your hypothesis about the influence of ENSO on Colorado Snow!!
# 
# Instructions for bootstrap:  Say there are N years with El Nino conditions. Instead of averaging the Loveland SWE in those N years, randomly grab N Loveland SWE values and take their average.  Then do this again, and again, and again 1000 times.  In the end you will end up with a distribution of SWE averages in the case of random sampling, i.e., the distribution you would expect if there was no physical relationship between Nino3.4 SST anomalies and Loveland Pass SWE.  
# 
# -Plot a histogram of this distribution and provide basic statistics describing this distribution (mean, standard deviation, minimum, and maximum).  
# 
# -Quantify the likelihood of getting your value of mean SWE by chance alone using percentiles of this bootstrapped distribution.  What is the probability that differences between the El Nino composite and all years occurred by chance? What is the probability that differences between the La Nina composite and all years occurred by chance?
# 
# Test the sensitivity of the results obtained in 2) by changing the number of bootstraps, the statistical significance level, or the definition of El Nino/La Nina (e.g., change the temperature threshold so that El Nino is defined using a 0.5 degree C temperature anomaly or a 3 degree C temperature anomaly).    In other words, TINKER and learn something about the robustness of your conclusions.  

# In[27]:


### Bootstrap!!  Generate random samples of size N_SWE_nino and N_SWE_nina.  Do it once to see if it works.
P_random=np.random.choice(data.BerthoudPass_April1SWE_inches,N_SWE_nino)
print(P_random)  ## LOOK AT YOUR DATA


# In[29]:


### Now Bootstrap Nbs times to generate a distribution of randomly selected mean SWE.
Nbs=1000
## initialize array
P_Bootstrap=np.empty((Nbs,N_SWE_nino)) #zeros seems to work too
## loop over to fill in array with randomly selected values
for ii in range(Nbs):
    P_Bootstrap[ii,:]=np.random.choice(data.BerthoudPass_April1SWE_inches,N_SWE_nino)

## Calculate the means of your randomly selected SWE values.
P_Bootstrap_mean=np.mean(P_Bootstrap,axis=1)
print(len(P_Bootstrap_mean))  ## check length to see if you averaged across the correct axis
print(np.shape(P_Bootstrap_mean)) ## another option to look at the dimensions of a variable

P_Bootstrap_mean_avg=np.mean(P_Bootstrap_mean)
print(P_Bootstrap_mean_avg)
P_Bootstrap_mean_std=np.std(P_Bootstrap_mean)
print(P_Bootstrap_mean_std)
P_Bootstrap_mean_min=np.min(P_Bootstrap_mean)
print(P_Bootstrap_mean_min)
P_Bootstrap_mean_max=np.max(P_Bootstrap_mean)
print(P_Bootstrap_mean_max)


# In[10]:


### Use matplotlib to plot a histogram of the bootstrapped means to compare to the conditioned SWE mean
binsize=0.1
min4hist=np.round(np.min(P_Bootstrap_mean),1)-binsize
max4hist=np.round(np.max(P_Bootstrap_mean),1)+binsize
nbins=int((max4hist-min4hist)/binsize)

plt.figure()
plt.hist(P_Bootstrap_mean,nbins,edgecolor='black')
plt.xlabel('Mean SWE (inches)');
plt.ylabel('Count');
plt.title('Bootstrapped Randomly Selected Mean SWE Values');


# In[11]:


## What is the probability that the snowfall was lower during El Nino by chance?
## Using Barnes equation (83) on page 15 to calculate probability using z-statistic
sample_mean=SWE_avg_nino
sample_N=1
population_mean=np.mean(P_Bootstrap_mean)
population_std=np.std(P_Bootstrap_mean)
xstd=population_std/np.sqrt(sample_N)
z_nino=(sample_mean-population_mean)/xstd

print(f'sample_mean - El Nino: {np.round(sample_mean,2)}')
print(f'population_mean: {np.round(population_mean,2)}')
print(f'population_std: {np.round(population_std,2)}')
print(f'Z-statistic (# standard errors that the sample mean deviates from the population mean: {np.round(z_nino,2)}')
prob=(1-stats.norm.cdf(np.abs(z_nino)))*100 ##this is a one-sided test
print(f'Probability happened by chance, one-tailed test (percent): {np.round(prob,0)}%')


# In[12]:


## What is the probability that the snowfall El Nino mean differs from the mean by chance?
## Using Barnes equation (83) on page 15 to calculate probability using z-statistic
sample_mean=SWE_avg_nino
sample_N=1
population_mean=np.mean(P_Bootstrap_mean)
population_std=np.std(P_Bootstrap_mean)
xstd=population_std/np.sqrt(sample_N)
z_nino=(sample_mean-population_mean)/xstd

print(f'sample_mean - El Nino: {np.round(sample_mean,2)}')
print(f'population_mean: {np.round(population_mean,2)}')
print(f'population_std: {np.round(population_std,2)}')
print(f'Z-statistic (# standard errors that the sample mean deviates from the population mean: {np.round(z_nino,2)}')

prob=(1-stats.norm.cdf(np.abs(z_nino)))*2*100 ##this is a two-sided test
print(f'Probability happened by chance, two-tailed test (percent): {np.round(prob,0)}%')


# In[13]:


## What is the probability that the snowfall was higher during La Nina just due to chance?
## Using Barnes equation (83) on page 15 to calculate probability using z-statistic
sample_mean=SWE_avg_nina
sample_N=1
population_mean=np.mean(P_Bootstrap_mean)
population_std=np.std(P_Bootstrap_mean)
xstd=population_std/np.sqrt(sample_N)
z_nina=(sample_mean-population_mean)/xstd

print(f'sample_mean - La Nina: {np.round(sample_mean,2)}')
print(f'population_mean: {np.round(population_mean,2)}')
print(f'population_std: {np.round(population_std,2)}')
print(f'Z-statistic (# standard errors that the sample mean deviates from the population mean: {np.round(z_nina,2)}')
prob=(1-stats.norm.cdf(np.abs(z_nina)))*100 ##this is a one-sided test
print(f'Probability happened by chance, one-tailed test (percent): {np.round(prob,0)}%')


# In[14]:


## What is the probability that the snowfall during La Nina differed just due to chance?
## Using Barnes equation (83) on page 15 to calculate probability using z-statistic
sample_mean=SWE_avg_nina
sample_N=1
population_mean=np.mean(P_Bootstrap_mean)
population_std=np.std(P_Bootstrap_mean)
xstd=population_std/np.sqrt(sample_N)
z_nina=(sample_mean-population_mean)/xstd

print(f'sample_mean - El Nino: {np.round(sample_mean,2)}')
print(f'population_mean: {np.round(population_mean,2)}')
print(f'population_std: {np.round(population_std,2)}')
print(f'Z-statistic (# standard errors that the sample mean deviates from the population mean: {np.round(z_nina,2)}')
prob=(1-stats.norm.cdf(np.abs(z_nina)))*2*100 ##this is a two-sided test
print(f'Probability happened by chance, two-tailed test (percent): {np.round(prob,0)}%')


# #### Maybe you want to set up the bootstrap in another way?? 
# Another bootstrapping approach is provided by Vineel Yettella (ATOC Ph.D. 2018).  
# Check these out and see what you find!!

# In[15]:


### Another bootstrapping strategy (provided by Vineel Yettella)
SWE = data['BerthoudPass_April1SWE_inches']
SWE_nino = data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches']

#We start by setting up a null hypothesis H0. 
#Our H0 will be that the difference in means of the two populations that the samples came from is equal to zero.
#We will use the bootstrap to test this null hypothesis.

#We next choose a significance level for the hypothesis test
alpha = 0.05

#All hypothesis tests need a test statistic.
#Here, we'll use the difference in sample means as the test statistic.
#create array to hold bootstrapped test statistic values
bootstrap_statistic = np.empty(10000)

#bootstrap 10000 times
for i in range(1,10000):
    
    #create a resample of SWE by sampling with replacement (same length as SWE)
    resample_original = np.random.choice(SWE, len(SWE), replace=True)
    
    #create a resample of SWE_nino by sampling with replacement (same length as SWE_nino)
    resample_nino = np.random.choice(SWE_nino, len(SWE_nino), replace=True)
    
    #Compute the test statistic from the resampled data, i.e., the difference in means
    bootstrap_statistic[i] = np.mean(resample_original) - np.mean(resample_nino)

#Let's plot the distribution of the test statistic
plt.figure()
plt.hist(bootstrap_statistic,[-5,-4,-3,-2,-1,0,1,2,3,4,5],edgecolor='black')
plt.xlabel('Difference in sample means')
plt.ylabel('Count')
plt.title('Bootstrap distribution of difference in sample means')

#Create 95% CI from the bootstrapped distribution. The upper limit of the CI is defined as the 97.5% percentile
#and the lower limit as the 2.5% percentile of the boostrap distribution, so that 95% of the 
#distribution lies within the two limits

CI_up = np.percentile(bootstrap_statistic, 100*(1 - alpha/2.0))
CI_lo = np.percentile(bootstrap_statistic, 100*(alpha/2.0))

print(CI_up)
print(CI_lo)

#We see that the confidence interval contains zero, so we fail to reject the null hypothesis that the difference
#in means is equal to zero


# ### Question 3: Do you get the same result when you use a t-test?
# 
# Check your assumptions for the t-test and understand what is "under the hood" of your python coding.

# In[16]:


## Apply a t-test to test the null hypothesis that the means of the two samples are the same 
## at the 95% confidence level.  Is this a one-sided or two-sided test??  Does it match what you got above??

## Calculate the t-statistic using the Barnes Notes - Compare a sample mean and a population mean.
## Barnes Eq. (96)
N=len(data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'])
print(f'N: {N}')
sample_mean=np.mean(data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'])
print(f'sample_mean: {np.round(sample_mean)}')
sample_std=np.std(data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'])
print(f'sample_std: {np.round(sample_std,2)}')
population_mean=np.mean(data['BerthoudPass_April1SWE_inches'])

## Using Barnes equation (96) to calculate probability using the t-statistic
t=(sample_mean-population_mean)/(sample_std/(np.sqrt(N-1)))
print(f'T-statistic: {np.round(t,2)}')
prob=(1-stats.t.cdf(t,N-1))*100
print(f'Probability (percent): {np.round(prob,0)}%')


# In[17]:


## Calculate the t-statistic using the Barnes Notes - Compare two sample means.  Equation (110)
## See page 26 of Chapter 1 of the Barnes notes for a worked example.

sampledata1=data['BerthoudPass_April1SWE_inches']
sampledata2=data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches']

N1=len(sampledata1)
N2=len(sampledata2)
print(f'N1={N1}, N2={N2}')
sample_mean1=np.mean(sampledata1)
sample_mean2=np.mean(sampledata2)
print(sample_mean1)
print(sample_mean2)
sample_std1=np.std(sampledata1)
sample_std2=np.std(sampledata2)
print(sample_std1)
print(sample_std2)

print("T-statistic using Barnes Eq. 109/Eq. 110:")
s=np.sqrt((N1*sample_std1**2+N2*sample_std2**2)/(N1+N2-2))
print(f's: {np.round(s,2)}')
tw=(sample_mean1-sample_mean2-0)/(s*np.sqrt(1/N1+1/N2))
print(f'tw: {np.round(tw,2)}')
prob=(1-stats.t.cdf(tw,N-1))*100
print(f'Probability (percent): {np.round(prob,0)}%')


# In[18]:


### Always try to code it yourself to understand what you are doing.
## Word to the wise - understand what is "under the hood" of your python function...
## Wait a second - What is that stats.ttest_ind function doing???  
# Check out the documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
# Review assumptions made with regard to the variances of the samples you are comparing...

print('Null Hypothesis: ENSO snow years have the same mean as the full record.')
t=stats.ttest_ind(data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'],data['BerthoudPass_April1SWE_inches'],equal_var=False)
#Note: When equal_var is false (defaults to true), you assume the underlying population variances are NOT equal 
## and this function then uses Welch's T-test
print(f't: {np.round(t.statistic,2)}')
print(f'pvalue: {np.round(t.pvalue,2)}')

######## example using python function = improved after discussions with Yu-Wen in office hours :)
print('Try using a ttest function from python - using the p-value')
#stats.ttest_ind(gts_1850_norm,gts_mem1_norm)

print(t.pvalue)

if t.pvalue < 0.05:
    print('Can reject the null hypthesis.')
else:
    print('Cannot reject the null hypthesis.')   
    
#### can also compare tstatistic to tcrit to evaluate statistical significance.
N=len(data[data.Nino34_anomaly_prevDec>1.0]['BerthoudPass_April1SWE_inches'])
tcrit=stats.t.ppf(0.975,N-1)
print(f'tcrit: {np.round(tcrit,2)}')

if np.abs(t.statistic) > np.abs(tcrit):
    print('Can reject the null hypthesis.')
else:
    print('Cannot reject the null hypthesis.')     
    


# ### SUMMARIZE WHAT YOU FOUND AND WHAT YOU LEARNED...  
# 
# Does ENSO affect total Colorado snow accumulation at Berthoud Pass, Colorado in the following winter? 

# In[ ]:




