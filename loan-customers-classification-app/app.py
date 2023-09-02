"""
Dashboard aims to....
"""
import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt
from streamlit.components import v1 as components
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from lime import lime_tabular

logo_image = Image.open("./data/logo.png")
st.set_page_config(
    page_title="CREDIT SCORING - DACHBOARD CLIENT SCORING",
    page_icon=logo_image,
    layout="wide",
)


@st.cache_data
@st.cache_resource
def load():
    """
    This functions aims to load data and models
    """
    model = joblib.load(open(os.path.join("./models/best_model.pkl"), "rb"))
    features = joblib.load(open(os.path.join("./data/training_features.pkl"), "rb"))
    dataframe = joblib.load(open(os.path.join("./data/full_data.pkl"), "rb"))

    return model, features, dataframe


loan_scoring_classifier, training_features, raw_data = load()

scaler = MinMaxScaler()
data = scaler.fit_transform(raw_data[training_features])
data = pd.DataFrame(data, index=raw_data.index, columns=training_features)
raw_data = raw_data.reset_index()
probas = loan_scoring_classifier.predict_proba(data)
raw_data["proba_true"] = probas[:, 0]
mean_score = raw_data["proba_true"].mean()

explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(data),
    feature_names=data.columns,
    # class_names=['bad', 'good'],
    mode="classification",
)


def plot_preds_proba(customer_id):
    """
    This functions aims plot income, annuities and credit vizuals
    """
    user_infos = {
        "Income": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_INCOME_TOTAL"
        ].values[0],
        "Credit": raw_data[raw_data["SK_ID_CURR"] == customer_id]["AMT_CREDIT"].values[
            0
        ],
        "Annuity": raw_data[raw_data["SK_ID_CURR"] == customer_id][
            "AMT_ANNUITY"
        ].values[0],
    }
    pred_proba_df = pd.DataFrame(
        {"Amount": user_infos.values(), "Operation": user_infos.keys()}
    )
    c = (
        alt.Chart(pred_proba_df)
        .mark_bar()
        .encode(x="Operation", y="Amount", color="Operation")
        .properties(width=330, height=310)
    )
    st.altair_chart(c)


def main():
    # main function
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        col2.image(logo_image, use_column_width=True)
        st.markdown("""---""")
        st.markdown(
            """
                        <h4 style='text-align: center; color: black;'> Wo we are? </h4>
                        """,
            unsafe_allow_html=True,
        )
        st.info("""We are a financial company that offers consumer credit""")
        st.markdown("""---""")
        cover_image = Image.open("./data/cover.jpg")
        st.sidebar.image(cover_image, use_column_width=True)

    tab1, tab2, tab3 = st.tabs(
        ["🏠 About this project", "📈 Make Predictions & Analyze", "🗃 Data Drift Reports"]
    )
    tab1.markdown("""---""")
    tab1.subheader("Credit Score")
    tab1.markdown(
        "This tool gives **guidance in credit granting decision** for our Relationship Managers. Based on customer's loan history and personnal informations, it predicts whether if he can refund a credit. It is based on one of the most powerful boosting algorith: **LightGBM**. \n To start, click on 'Make predictions & Analyze' at the top of the page. "
    )
    tab1.markdown("""---""")

    with tab1.markdown("**Project Lifecycle**"):
        col1, col2 = st.columns(2)
        col1.info("**About the compagny**")
        home_image = Image.open("./data/home_credit.jpeg")
        col1.image(home_image, use_column_width=True)
        col1.markdown(
            "<p style='text-align: justify;'>Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities. While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.</p>",
            unsafe_allow_html=True,
        )

        lifecycle_schema = Image.open("./data/lifecycle.png")
        col2.info("**Project Lifecycle**")
        col2.image(lifecycle_schema, use_column_width=True)

    with tab2.subheader("Loan Scoring Model"):
        with st.form(key="myform"):
            # user_liste = data.index
            # user_id_value = st.selectbox('Select customer id', user_liste)
            user_id_value = st.number_input("Select customer id", min_value=100001)
            submit_button = st.form_submit_button(label="Show")

            if submit_button:
                if isinstance(user_id_value, int) and user_id_value in data.index:
                    st.write("Customer selected : ", user_id_value)
                    col1, col2 = st.columns(2)
                    # data = data.reset_index()
                    user = data[data.index == int(user_id_value)]
                    # prediction = loan_scoring_classifier.predict(user)[0]
                    probas_user = loan_scoring_classifier.predict_proba(user)
                    probabilities = dict(
                        zip(
                            loan_scoring_classifier.classes_,
                            np.round(probas_user[0], 3),
                        )
                    )

                    # display results
                    with col1:
                        st.info("Customer Infos")
                        # st.text(f'User Id : {user_id_value}')
                        user_infos = raw_data[raw_data["SK_ID_CURR"] == user_id_value]

                        # st.write('Age', int(user_infos["DAYS_BIRTH"]/ -365))
                        # st.write('Sex', user_infos["CODE_GENDER"].item())
                        # st.write('Status', user_infos["NAME_FAMILY_STATUS"].item())
                        # st.write('Age', int(user_infos["DAYS_BIRTH"]/ -365))
                        # user_infos = user_infos[[]]
                        dict_infos = {
                            "Age": int(user_infos["DAYS_BIRTH"] / -365),
                            "Gender": user_infos["CODE_GENDER"]
                            .replace(["F", "M"], ["Female", "Male"])
                            .item(),
                            "Status": user_infos["NAME_FAMILY_STATUS"].item(),
                            "Education": user_infos["NAME_EDUCATION_TYPE"].item(),
                            "Employment_Seniority": int(
                                user_infos["DAYS_EMPLOYED"].values / -365
                            ),
                            "Income_Type": user_infos["NAME_INCOME_TYPE"].item(),
                            "Income_Ammount": user_infos["AMT_INCOME_TOTAL"].item(),
                        }
                        st.write(dict_infos)

                        # st.info('Results')
                        st.markdown("""---""")
                        st.info("Loan History")
                        dict_infos = {
                            "Type contrat": user_infos["NAME_CONTRACT_TYPE"].item(),
                            "Montant_credit": user_infos["AMT_CREDIT"].item(),
                            "Annuites": user_infos["AMT_ANNUITY"].item(),
                        }
                        user = data[data.index == int(user_id_value)]
                        st.write(dict_infos)
                        # st.metric(label='Accuracy', value='', delta='1.6')
                        st.markdown("""---""")
                        st.info("Credit Score")
                        # c1, c2, c3, c4, c5 = st.columns(5)
                        if round(probabilities[0] * 100, 2) > 60:
                            st.metric(
                                "High Score",
                                value=round(probabilities[0] * 100, 2),
                                delta=f"{round((probabilities[0]-0.6)*100,2)}",
                            )

                            st.success(
                                "This customer is a potential refunder", icon="✅"
                            )
                        elif 50 < round(probabilities[0] * 100, 2) < 60:
                            st.metric(
                                "Medium Score",
                                value=round(probabilities[0] * 100, 2),
                                delta=f"{round((probabilities[0]-0.6)*100,2)}",
                            )

                            st.warning(
                                "This customer may have difficulties in refunding",
                                icon="⚠️",
                            )
                        else:
                            st.metric(
                                "Low Score",
                                value=round(probabilities[0] * 100, 2),
                                delta=f"{round((probabilities[0]-0.6)*100,2)}",
                            )
                            st.error("This customer can not refund", icon="🚨")
                    with col2:
                        st.info("Features contribution")
                        exp = explainer.explain_instance(
                            data_row=data.loc[user_id_value],
                            predict_fn=loan_scoring_classifier.predict_proba,
                        )
                        components.html(exp.as_html(), height=550)
                        st.markdown("""---""")
                        plot_preds_proba(user_id_value)
                else:
                    st.error("Please, enter a valid customer id.", icon="🚨")

            else:
                st.markdown("""---""")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Quick Tutorial**")
                    st.info(
                        """
                        1. Select or edit **Customer ID**.
                        2. Click **Show Infos** button.
                        4. Analyze user informations and prediction results
                        """,
                    )
                    st.markdown("**Few Tips**")
                    st.success(
                        """
                        1. Consider all the informations, use the vizual oh the bottom right.
                        2. Use Credit Score wisely, delta becomes green for potential refunder.
                        4. Keep an eye on the data drift report is essential.
                        """,
                    )

                with col2:
                    st.markdown("**Global Explainability of the model**")
                    st.image(
                        Image.open("./data/explainability.png"), use_column_width=True
                    )

    with tab3.subheader("Data Drift Report"):
        report = open("./data/data_drift_report.html", "r", encoding="utf-8")
        source_code = report.read()
        components.html(source_code, height=1500)


if __name__ == "__main__":
    main()
