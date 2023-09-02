from app import training_features, raw_data


def test_training_features():
    assert len(training_features) == 797


def test_personal_infos_features():
    infos_features = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
        "CODE_GENDER",
        "NAME_FAMILY_STATUS",
        "NAME_EDUCATION_TYPE",
        "DAYS_EMPLOYED",
        "NAME_INCOME_TYPE",
        "AMT_INCOME_TOTAL",
        "NAME_CONTRACT_TYPE",
    ]

    compare = all([item in raw_data.columns for item in infos_features])
    assert compare == True
