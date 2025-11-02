import pandas as pd
import numpy as np

# Fonction de nettoyage existante
def clean_studentInfo(df):
    regions = df[df['imd_band'].isnull()]['region'].unique()
    for region in regions:
        mode_value = df[df['region'] == region]['imd_band'].mode()
        if not mode_value.empty:
            df.loc[(df['imd_band'].isnull()) & (df['region'] == region), 'imd_band'] = mode_value[0]
    return df

def clean_studentAssessment(df):
    df['score'] = df['score'].fillna(0)
    return df

def clean_studentRegistration(df):
    return df.drop(columns=['date_registration'])

def clean_all(studentInfo, studentAssessment, studentRegistration, studentVle, courses, vle, assessments):
    return {
        "studentInfo": clean_studentInfo(studentInfo),
        "studentAssessment": clean_studentAssessment(studentAssessment),
        "studentRegistration": clean_studentRegistration(studentRegistration),
        "studentVle": studentVle.fillna({'sum_click': 0}),
        "courses": courses,
        "vle": vle,
        "assessments": assessments
    }