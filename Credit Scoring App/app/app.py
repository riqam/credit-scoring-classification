import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import pickleshare
import sklearn
import seaborn as sns
import streamlit as st
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import utils as utils
from xgboost import XGBClassifier
import joblib

model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
standardizer = joblib.load('standardizer.pkl')

CONFIG_DATA = utils.config_load()

def transform_resp(resp):
    def yes_no(column):
        if resp[column] == 'Yes':
            return 1
        else:
            return 0

    output = {
        'RevolvingUtilizationOfUnsecuredLines': resp['RevolvingUtilizationOfUnsecuredLines'],
        'age': resp['age'],
        'NumberOfTime30-59DaysPastDueNotWorse': resp['NumberOfTime30-59DaysPastDueNotWorse'],
        'DebtRatio': resp['DebtRatio'],
        'MonthlyIncome': resp['MonthlyIncome'],
        'NumberOfOpenCreditLinesAndLoans': resp['NumberOfOpenCreditLinesAndLoans'],
        'NumberOfTimes90DaysLate': resp['NumberOfTimes90DaysLate'],
        'NumberRealEstateLoansOrLines': resp['NumberRealEstateLoansOrLines'],
        'NumberOfTime60-89DaysPastDueNotWorse': resp['NumberOfTime60-89DaysPastDueNotWorse'],
        'NumberOfDependents': resp['NumberOfDependents']
    }

    return output


st.set_page_config(page_title='Credit Score App', page_icon='💰', layout='wide',
                   initial_sidebar_state='auto', menu_items={
                        'Get Help': None,
                        'Report a bug': 'https://github.com/riqam/credit-scoring/issues',
                        'About': '''
                        This app was made by **M. Rifqi Akram** and its purpose is to showcase how a Credit Score Evaluation work in the fictional bank _Bankio_. 
                        
                        This evaluation is using a Machine Learning model, and you can learn more about how the model work and how I got here by going to the GitHub [repository](https://github.com/riqam/credit-scoring).
                        If you are interested in Data Science you can see follow my work through my LinkedIn [M. Rifqi Akram](https://www.linkedin.com/in/m-rifqi-akram/).
                        '''
     })
    
st.title('Credit Score Analysis')
st.caption('Made by M. Rifqi Akram © 2024')

st.markdown('''
            This is a mock-up intended for information only, if you wish to learn more about the model behind this please go to the GitHub [repository](https://github.com/riqam/credit-scoring).
            On the left, there's a sidebar (click ` > ` if you don't see it). Where you can fill out a form - _this data is not saved_ - to see how each piece of information you provided impacts debtor's credit score.
            
            On the top right corner, there is a ` ⁝ ` button if you want to know more information about this App or if you want to report a bug that you find in this App.
            
            I created two profiles of random people as examples to fill out the form with their information! And don't forget to click the button `Calculate scoring!` on the sidebar.
''')

name_default = ''
age_default = 0
MonthlyIncome_default = 0.0000
total_balance_credit_card_default = 0.0000
credit_limits_default = 0.0000
debtor_expenses_default = 0.0000
NumberOfOpenCreditLinesAndLoans_default = 0
NumberOfTime30_59DaysPastDueNotWorse_default = 0
NumberOfTime60_89DaysPastDueNotWorse_default = 0
NumberOfTimes90DaysLate_default = 0
NumberRealEstateLoansOrLines_default = 0
NumberOfDependents_default = 0

profile = st.radio('Choose a profile:', options=['Rifqi', 'Akram'], horizontal=True)

if profile == 'Rifqi':
    name_default = 'Rifqi'
    age_default = 45
    MonthlyIncome_default = 9120.0000
    total_balance_credit_card_default = 766126609.0000
    credit_limits_default = 1000000000.0000
    debtor_expenses_default = 7323.19701648
    NumberOfOpenCreditLinesAndLoans_default = 13
    NumberOfTime30_59DaysPastDueNotWorse_default = 2
    NumberOfTime60_89DaysPastDueNotWorse_default = 0
    NumberOfTimes90DaysLate_default = 0
    NumberRealEstateLoansOrLines_default = 6
    NumberOfDependents_default = 2

elif profile == 'Akram':
    name_default = 'Akram'
    age_default = 40
    MonthlyIncome_default = 2600.0000
    total_balance_credit_card_default = 957151019.0000
    credit_limits_default = 1000000000.0000
    debtor_expenses_default = 316.8800
    NumberOfOpenCreditLinesAndLoans_default = 4
    NumberOfTime30_59DaysPastDueNotWorse_default = 0
    NumberOfTime60_89DaysPastDueNotWorse_default = 0
    NumberOfTimes90DaysLate_default = 0
    NumberRealEstateLoansOrLines_default = 0
    NumberOfDependents_default = 1

with st.sidebar:
    st.header('Credit Score Form')
    st.text_input("What is the debtor's name?", value=name_default)
    age = st.slider("What is the debtor's age?", min_value=18, max_value=100, step=1, value=age_default)
    MonthlyIncome = st.number_input("What is the debtor's Monthly Income?", step=100000.0000, value=MonthlyIncome_default)
    total_balance_credit_card = st.number_input("How much is the total balance on the debtor's credit cards and personal lines of credit?", step=100000.0000, value=total_balance_credit_card_default)
    credit_limits = st.number_input("How much credit limit does the debtor have?", step=100000.0000, value=credit_limits_default)
    debtor_expenses = st.number_input("How much money do debtors spend every month?", step=100000.0000, value=debtor_expenses_default)
    NumberOfOpenCreditLinesAndLoans = st.number_input('How many open loans and Lines of credit do debtors have?', step=1, value=NumberOfOpenCreditLinesAndLoans_default)
    NumberOfTime30_59DaysPastDueNotWorse = st.number_input('How many times has a debtor been 30-59 days past due but not worse in the last 2 years?', step=1, value=NumberOfTime30_59DaysPastDueNotWorse_default)
    NumberOfTime60_89DaysPastDueNotWorse = st.number_input('How many times has a debtor been 60-89 days delinquent but not worse in the last 2 years?', step=1, value=NumberOfTime60_89DaysPastDueNotWorse_default)
    NumberOfTimes90DaysLate = st.number_input('How many times has the debtor been 90 days or more delinquent?', step=1, value=NumberOfTimes90DaysLate_default)
    NumberRealEstateLoansOrLines = st.number_input('How many mortgage and real estate loans include a home equity line of credit?', step=1, value=NumberRealEstateLoansOrLines_default)
    NumberOfDependents = st.number_input('How many dependents does the debtor have?', step=1, value=NumberOfDependents_default)

    run = st.button("Calculate scoring!", type="primary")

st.header('Credit Score Results')

col1, col2 = st.columns([3, 2])

with col2:
    x1 = [0, 6, 0]
    x2 = [0, 3, 0]
    y = ['0', '1', '2']

    f, ax = plt.subplots(figsize=(5,2))

    p1 = sns.barplot(x=x1, y=y, color='#3EC300')
    p1.set(xticklabels=[], yticklabels=[])
    p1.tick_params(bottom=False, left=False)
    p2 = sns.barplot(x=x2, y=y, color='#FF331F')
    p2.set(xticklabels=[], yticklabels=[])
    p2.tick_params(bottom=False, left=False)

    plt.text(1.5, 1.1, "POOR", horizontalalignment='center', size='medium', color='white', weight='semibold')
    plt.text(4.5, 1.1, "GOOD", horizontalalignment='center', size='medium', color='white', weight='semibold')

    ax.set(xlim=(0, 6))
    sns.despine(left=True, bottom=True)
    
    figure = st.pyplot(f)

with col1:

    placeholder = st.empty()

    if run:
        resp = {
            'RevolvingUtilizationOfUnsecuredLines': total_balance_credit_card/credit_limits,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': NumberOfTime30_59DaysPastDueNotWorse,
            'DebtRatio': debtor_expenses/MonthlyIncome,
            'MonthlyIncome': MonthlyIncome,
            'NumberOfOpenCreditLinesAndLoans': NumberOfOpenCreditLinesAndLoans,
            'NumberOfTimes90DaysLate': NumberOfTimes90DaysLate,
            'NumberRealEstateLoansOrLines': NumberRealEstateLoansOrLines,
            'NumberOfTime60-89DaysPastDueNotWorse': NumberOfTime60_89DaysPastDueNotWorse,
            'NumberOfDependents': NumberOfDependents,
        }
        
        output = transform_resp(resp)
        output = pd.DataFrame(output, index=[0])
        output.loc[:,:] = standardizer.transform(output)
        
        proba = model.predict_proba(output)[:, 1]
        
        if proba < 0.8:
            st.success(' The credit score is **GOOD**!', icon="✅")
            t1 = plt.Polygon([[4.5, 0.5], [5.0, 0], [4.0, 0]], color='black')
            st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is **_low_**.')
        elif proba >= 0.8:
            st.error(" WARNING! The credit score is **POOR**.", icon="❌")
            t1 = plt.Polygon([[1.5, 0.5], [2.0, 0], [1.0, 0]], color='black')
            st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is **_high_**.')

        plt.gca().add_patch(t1)
        figure.pyplot(f)
        prob_fig, ax = plt.subplots()