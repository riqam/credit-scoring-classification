def transform_resp(resp):
    def yes_no(column):
        if resp[column] == 'Yes':
            return 1
        else:
            return 0

    output = {
        'age': resp['age'],
        'RevolvingUtilizationOfUnsecuredLines': resp['RevolvingUtilizationOfUnsecuredLines'],
        'NumberOfTime30-59DaysPastDueNotWorse': resp['NumberOfTime30-59DaysPastDueNotWorse'],
        'DebtRatio': resp['DebtRatio'],
        'MonthlyIncome': resp['MonthlyIncome'],
        'Credit_Utilization_Ratio': resp['credit_card_ratio'],
        'NumberOfOpenCreditLinesAndLoans': resp['NumberOfOpenCreditLinesAndLoans'],
        'NumberOfTimes90DaysLate': resp['NumberOfTimes90DaysLate'],
        'NumberRealEstateLoansOrLines': resp['NumberRealEstateLoansOrLines'],
        'NumberOfTime60-89DaysPastDueNotWorse': resp['NumberOfTime60-89DaysPastDueNotWorse'],
        'NumberOfDependents': resp['NumberOfDependents']
    }

    return output