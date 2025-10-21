import pandas as pd


studentRegistration = pd.read_csv('../data/studentRegistration.csv')
studentInfo = pd.read_csv('../data/studentInfo.csv')
studentVle = pd.read_csv('../data/vle.csv')
studentAssessment = pd.read_csv('../data/studentAssessment.csv')
courses = pd.read_csv('../data/courses.csv')
vle = pd.read_csv('../data/vle.csv')
assessments = pd.read_csv('../data/assessments.csv')

def clean_studentInfo(df):
    # Impute imd_band by region
    regions = df[df['imd_band'].isnull()]['region'].unique()
    for region in regions:
        mode_value = df[df['region'] == region]['imd_band'].mode()
        if not mode_value.empty:
            df.loc[(df['imd_band'].isnull()) & (df['region'] == region), 'imd_band'] = mode_value[0]
    return df

def clean_studentAssessment(df):
    # Replace missing scores with 0 (considered fail)
    df['score'] = df['score'].fillna(0)
    return df

def clean_studentRegistration(df):
    # remove the column 'date_registration' due to high missing values
    df = df.drop(columns=['date_registration'])
    return df

def clean_all():
    return {
        "studentInfo": clean_studentInfo(studentInfo),
        "studentAssessment": clean_studentAssessment(studentAssessment),
        "studentRegistration": clean_studentRegistration(studentRegistration),
        "studentVle": studentVle.dropna(),
        "courses": courses,
        "vle": vle,
        "assessments": assessments
    }

