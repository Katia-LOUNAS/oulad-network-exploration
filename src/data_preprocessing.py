import pandas as pd 
from pathlib import Path

# =============================================
# This file loads and prepares the OULAD dataset
# for the part based on graph for students.
#
# We merge the main tables and build a clean dataset
# with information and activity of students.
# =============================================

def load_raw_data(data_dir, debug=False):
    """load all csv files present for the project oulad"""
    data_path = Path(data_dir)

    dfs = {
        "studentInfo": pd.read_csv(data_path / "studentInfo.csv"),
        "studentRegistration": pd.read_csv(data_path / "studentRegistration.csv"),
        "studentVle": pd.read_csv(data_path / "studentVle.csv"),
        "vle": pd.read_csv(data_path / "vle.csv"),
        "courses": pd.read_csv(data_path / "courses.csv"),
        "assessments": pd.read_csv(data_path / "assessments.csv"),
        "studentAssessment": pd.read_csv(data_path / "studentAssessment.csv"),
    }

    if debug:
        print("All CSV files loaded successfully!")
        for name, df in dfs.items():
            print(f"{name:20s}: {df.shape[0]} rows, {df.shape[1]} columns")

    return dfs


def build_student_table(dfs, debug=False):
    """
    For having a df with all student information
    -> for that we merge studentInfo, studentRegistration and courses tables
    => so we have : each row = one student registered in one module and presentation
    """
    student_data = pd.merge(
        dfs["studentRegistration"],
        dfs["studentInfo"],
        on=["id_student", "code_module", "code_presentation"],
        how="inner",
        validate="one_to_one"
    )

    # we add course information
    student_data = pd.merge(
        student_data,
        dfs["courses"],
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one"
    )
    if debug:
        print(f"student_data created: {student_data.shape[0]} rows\n")
        print(student_data.head())
    
    return student_data


def build_student_activity_matrix(dfs, module, presentation, debug=False):
    """
    part for build a matrix of student activity (number of clicks per VLE resource).
    -> useful for the construction of the graph 
    Each row = one student
    Each column = one resource (id_site)
    Value = total number of clicks (sum_click)

    Parameters
    ----------
    module : Code of the module (ex: 'AAA')
    presentation : Code of the presentation (ex: '2013J')

    Returns
    -------
    tableau : students x resources for having the info of nb click 
    """

    # we filter only the selected module and presentation
    df = dfs["studentVle"]
    df = df[(df["code_module"] == module) & (df["code_presentation"] == presentation)]

    # create pivot table => so the matrix
    pivot = df.pivot_table(
        index="id_student",
        columns="id_site",
        values="sum_click",
        aggfunc="sum",
        fill_value=0
    )

    if debug:
        print(f" activity matrix created for {module}_{presentation}: {pivot.shape}\n")
        print(pivot)
    
    return pivot


def build_student_assessment_table(dfs, debug=False):
    """
    merge of assessment and studentAssessment tables
    => eahc row = one student and one assessment (with score, weight, ...)
    -> useful for the analysing and not for the construction of the graph 
    """
    df = pd.merge(
        dfs["assessments"],
        dfs["studentAssessment"],
        on="id_assessment",
        how="inner",
        validate="one_to_many"
    )

    # we remove old transferred grades (is_banked == 1) -> he didn't really work this year
    df = df[df["is_banked"] == 0]

    if debug:
        print(f" student_assessment_data created: {df.shape[0]} rows\n")
        print(df)

    return df

def normalize_rows_max(X):
    """
    just simple normalization: divide each row by its max.
    -> reduces the effect of very big click counts.
    """
    X = X.copy()
    row_max = X.max(axis=1).replace(0, 1)
    X = X.div(row_max, axis=0)
    return X


def load_and_prepare_oulad(data_dir, module, presentation, debug=False):
    """
    pipeline for load and prepare the data.

    1. Load raw CSVs
    2. Build student info table
    3. Build activity matrix
    4. Build assessment table
    """
    dfs = load_raw_data(data_dir)
    student_data = build_student_table(dfs, debug=debug) # all info about sutdent and courses 
    activity_matrix = build_student_activity_matrix(dfs, module, presentation, debug=debug) # matrice for 
    assessment_data = build_student_assessment_table(dfs, debug=debug) # link eval and mark

    return student_data, activity_matrix, assessment_data

