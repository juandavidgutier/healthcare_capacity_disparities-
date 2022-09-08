# -*- coding: utf-8 -*-
"""
script to reproduce the results of causal inference of the paper:
"Impact of healthcare capacity disparities in the COVID-19 vaccination 
coverage in the United States: A cross-sectional study"

"""

# importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
import econml
from econml.dml import DML, LinearDML, SparseLinearDML, NonParamDML, CausalForestDML
from econml.dr import DRLearner, ForestDRLearner, SparseLinearDRLearner
from econml.orf import DROrthoForest, DMLOrthoForest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from econml.inference import BootstrapInference
import numpy as np, scipy.stats as st
import scipy.stats as stats
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from econml.score import RScorer
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs
from sklearn.model_selection import train_test_split


# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

######
#import data
data = pd.read_csv("D:/clases/UDES/articulo CI Diego/Datasets/dataset_covid.csv", encoding='latin-1') 
data = data.dropna()

Y = data.low_vaccination_rate.to_numpy() ##low_vaccination_rate = Vaccination rate < 50%
T = data.RCHSI.to_numpy()
W = data[['SVI', 'HACBI']].to_numpy().reshape(-1, 2)
X = data[['Vaccine_hesitancy']].to_numpy().reshape(-1, 1)

X_train, X_val, T_train, T_val, Y_train, Y_val, W_train, W_val = train_test_split(X, T, Y, W, test_size=.4)

warnings.filterwarnings('ignore') 

reg1 = lambda: GradientBoostingClassifier()
reg2 = lambda: GradientBoostingRegressor()

models = [
        ('ldml', LinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                             linear_first_stages=False, cv=3, random_state=123)),
        ('sldml', SparseLinearDML(model_y=reg1(), model_t=reg2(), discrete_treatment=False,
                                    featurizer=PolynomialFeatures(degree=3, include_bias=False),
                                    linear_first_stages=False, cv=3, random_state=123)),
        ('dml', DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                             featurizer=PolynomialFeatures(degree=3, include_bias=False),
                             linear_first_stages=False, cv=3, random_state=123)),
        ('forest', CausalForestDML(model_y=reg1(), model_t=reg2(), 
                                    featurizer=PolynomialFeatures(degree=3, include_bias=False), 
                                    discrete_treatment=False, cv=3, random_state=123)),
          ]


def fit_model(name, model):
    return name, model.fit(Y_train, T_train, X=X_train, W=W_train)

models = Parallel(n_jobs=-1, verbose=1, backend="threading")(delayed(fit_model)(name, mdl) for name, mdl in models)

#Choose model with highest RScore
scorer = RScorer(model_y=reg1(), model_t=reg2(),
                 discrete_treatment=False, cv=10,
                 mc_iters=10, mc_agg='median')

scorer.fit(Y_val, T_val, X=X_val, W=W_val)

rscore = [scorer.score(mdl) for _, mdl in models]
print(rscore)#best model DML


#Step 1: Model the causal mechanism
model_RCHSI=CausalModel(
        data = data,
        treatment=['RCHSI'],
        outcome=['low_vaccination_rate'],
        graph= """graph[directed 1 node[id "RCHSI" label "RCHSI"]
                    node[id "low_vaccination_rate" label "low_vaccination_rate"]
                    node[id "SVI" label "SVI"]
                    node[id "HACBI" label "HACBI"]
                    node[id "Hesitant" label "Hesitant"]
                    edge[source "SVI" target "RCHSI"]
                    edge[source "SVI" target "low_vaccination_rate"]
                    edge[source "HACBI" target "RCHSI"]
                    edge[source "HACBI" target "low_vaccination_rate"]
                    edge[source "SVI" target "HACBI"]
			  edge[source "HACBI" target "SVI"]
                    edge[source "SVI" target "Hesitant"]
                    edge[source "HACBI" target "Hesitant"]
                    edge[source "RCHSI" target "Hesitant"]                           
                    edge[source "RCHSI" target "low_vaccination_rate"]
                    edge[source "Hesitant" target "low_vaccination_rate"]
                    ]"""
                    )
    
#view model 
#model_RCHSI.view_model()

#Step 2: Identifying effects
identified_estimand_RCHSI = model_RCHSI.identify_effect(proceed_when_unidentifiable=False)
print(identified_estimand_RCHSI)

#Step 3: Estimating effects
estimate_RCHSI =  DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                            featurizer=PolynomialFeatures(degree=3, include_bias=False),
                            linear_first_stages=False, cv=3, random_state=123)

estimate_RCHSI = estimate_RCHSI.dowhy

# fit the CATE model
estimate_RCHSI.fit(Y=Y, T=T, X=X, W=W, inference='bootstrap')  

# predict effect for each sample X
estimate_RCHSI.effect(X)

# ate
ate_RCHSI = estimate_RCHSI.ate(X) 
print(ate_RCHSI)

# confidence interval of ate
ci_RCHSI = estimate_RCHSI.ate_interval(X) 
print(ci_RCHSI)



#Step 4: Refute the effect
#with random common cause
random_RCHSI = estimate_RCHSI.refute_estimate(method_name="random_common_cause", random_state=123)
print(random_RCHSI)


##with add unobserved common cause
unobserved_RCHSI = estimate_RCHSI.refute_estimate(method_name="add_unobserved_common_cause", 
                                                       confounders_effect_on_outcome="binary_flip", 
                                                       random_state=123)
print(unobserved_RCHSI)


##with bootstrap_refuter
boost_RCHSI = estimate_RCHSI.refute_estimate(method_name="bootstrap_refuter", random_state=123)                                                     
print(boost_RCHSI)


#with replace a random subset of the data
subset_RCHSI = estimate_RCHSI.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8, num_simulations=3)
print(subset_RCHSI)

#with placebo 
placebo_RCHSI = estimate_RCHSI.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", num_simulations=3)
print(placebo_RCHSI)





#CATE
#range of Vaccine_hesitancy
min_Hesitant = 0.33
max_Hesitant = 0.60
delta = (max_Hesitant - min_Hesitant) / 100
X_test = np.arange(min_Hesitant, max_Hesitant + delta - 0.001, delta).reshape(-1, 1)

# Calculate marginal treatment effects
treatment_effects = estimate_RCHSI.const_marginal_effect(X_test)

# Calculate default (95%) marginal confidence intervals for the test data
te_upper, te_lower = estimate_RCHSI.const_marginal_effect_interval(X_test)

est2_RCHSI =  DML(model_y=reg1(), model_t=reg2(), model_final=LassoCV(), discrete_treatment=False,
                        featurizer=PolynomialFeatures(degree=3, include_bias=False),
                        linear_first_stages=False, cv=3, random_state=123)

est2_RCHSI.fit(Y=Y, T=T, X=X, inference="bootstrap")

treatment_effects2 = est2_RCHSI.effect(X_test)
te_lower2_cons, te_upper2_cons = est2_RCHSI.effect_interval(X_test)


#Supplementary Figure 1
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  + geom_line() 
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Vaccine_hesitancy', y='Effect of RCHSI  on Vaccination rate <50% (%)')
  
)  







