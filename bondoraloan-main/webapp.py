import streamlit as st
import pandas as pd
import numpy as np
import warnings
import base64
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from PIL import Image
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

warnings.filterwarnings("ignore")
st.write("""
# Bondora Classifier
### Here we go!
""")
Data=pd.read_csv("finalData.csv")
def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

im=Image.open("graph.jpg")
st.sidebar.image(im)
st.sidebar.header("Borrower Input data")
st.sidebar.write("*Kindly answer the following questions to check the eligibility of the applied loan's :*")
def InputData():


    
    Language=selectL=st.sidebar.selectbox("Select language:",["English","Estonian","Finnish","German","Russian","Slovakian","Spanish","Other"])
    if(selectL=="English"): Language=0
    elif(selectL=="Estonian"): Language=1
    elif(selectL=="Finnish"): Language=2
    elif(selectL=="German"): Language=3
    elif(selectL=="Other"): Language=4
    elif(selectL=="Russian"): Language=5
    elif(selectL=="Slovakian"): Language=6
    else: Language =7

    selectCountry=selectL=st.sidebar.selectbox("Select Country:",["EE","ES","FI","SK"])
    if(selectCountry=="EE"): selectCountry=0
    elif(selectCountry=="ES"): selectCountry=1
    elif(selectCountry=="FI"): selectCountry=2
    else: selectCountry=3


    selectNewCreditCustomer = selectL = st.sidebar.selectbox("Select CreditCustomer:", ["New_credit_Customer", "Existing_credit_customer"])
    if (selectNewCreditCustomer == "New_credit_Customer"):
        selectNewCreditCustomer = 0
    else:
        selectNewCreditCustomer = 1

    selectRestructured = selectL = st.sidebar.selectbox("Select Restructured:", ["No", "Yes"])
    if (selectRestructured == "No"):
        selectRestructured = 0
    else:
        selectRestructured = 1


    selectVerificationType = selectL = st.sidebar.selectbox("Select VerificationType:",
                                   ["Income_expenses_verified", "Income_unverified", "Income_verified", "Income_unverified_crossref_phone", "Not_set"])
    if (selectVerificationType == "Income_expenses_verified"):
        selectVerificationType = 0
    elif (selectVerificationType == "Income_unverified"):
        selectVerificationType = 1
    elif (selectVerificationType == "Income_verified"):
        selectVerificationType = 2
    elif (selectVerificationType == "Income_unverified_crossref_phone"):
        selectVerificationType = 3
    else:
        selectVerificationType = 4

    selectGender = selectL = st.sidebar.selectbox("Select Gender:",
                                                            ["Male", "Woman","Undefined"])
    if (selectGender == "Male"):
        selectGender = 0
    elif (selectGender == "Woman"):
        selectGender = 1
    else:
        selectGender = 2

    selectUseOfLoan = selectL = st.sidebar.selectbox("Select UseOfLoan:",
                                                            ["Not_set",
                                                             "Other", "Home_improvement",
                                                             "Loan_consolidation", "Vehicle","Business","Travel","Health","Education","Real_estate",
                                                             "Purchase_of_machinery_equipment","Other_business","Accounts_receivable_financing",
                                                             "Working_capital_financing","Acquisition_of_real_estate","Acquisition_of_stocks","Construction_finance"])
    if (selectUseOfLoan == "Not_set"):
        selectUseOfLoan = 0
    elif (selectUseOfLoan == "Other"):
        selectUseOfLoan = 1
    elif (selectUseOfLoan == "Home_improvement"):
        selectUseOfLoan = 2
    elif (selectUseOfLoan == "Loan_consolidation"):
        selectUseOfLoan = 3
    elif (selectUseOfLoan == "Vehicle"):
        selectUseOfLoan = 4
    elif (selectUseOfLoan == "Business"):
        selectUseOfLoan = 5
    elif (selectUseOfLoan == "Travel"):
        selectUseOfLoan = 6
    elif (selectUseOfLoan == "Health"):
        selectUseOfLoan = 7
    elif (selectUseOfLoan == "Education"):
        selectUseOfLoan = 8
    elif (selectUseOfLoan == "Real_estate"):
        selectUseOfLoan = 9
    elif (selectUseOfLoan == "Purchase_of_machinery_equipment"):
        selectUseOfLoan = 10
    elif (selectUseOfLoan == "Other_business"):
        selectUseOfLoan = 11
    elif (selectUseOfLoan == "Accounts_receivable_financing"):
        selectUseOfLoan = 12
    elif (selectUseOfLoan == "Working_capital_financing"):
        selectUseOfLoan = 13
    elif (selectUseOfLoan == "Acquisition_of_real_estate"):
        selectUseOfLoan = 14
    elif (selectUseOfLoan == "Acquisition_of_stocks"):
        selectUseOfLoan = 15
    else:
        selectUseOfLoan = 16

    selectEducation = selectL = st.sidebar.selectbox("Select Education:",
                               ["Secondary education", "Higher education", "Vocational education", "Basic education", "Primary education","unkown"])
    if (selectEducation == "Secondary education"):
        selectEducation = 0
    elif (selectEducation == "Higher education"):
        selectEducation = 1
    elif (selectEducation == "Vocational education"):
        selectEducation = 2
    elif (selectEducation == "Basic education"):
        selectEducation = 3
    elif (selectEducation == "Primary education"):
        selectEducation = 4
    else:
         selectEducation = 5

    selectMaritalStatus = selectL = st.sidebar.selectbox("Select MaritalStatus:",
                                                            ["Single","Married","Cohabitant", "Divorced","Widow","unkown","Not_specified"])
    if (selectMaritalStatus == "Single"):
        selectMaritalStatus = 0
    elif (selectMaritalStatus == "Married"):
        selectMaritalStatus = 1
    elif (selectMaritalStatus == "Cohabitant"):
        selectMaritalStatus = 2
    elif (selectMaritalStatus == "Divorced"):
        selectMaritalStatus = 3
    elif (selectMaritalStatus == "Widow"):
        selectMaritalStatus = 4
    elif (selectMaritalStatus == "unkown"):
        selectMaritalStatus = 5
    else:
        selectMaritalStatus = 6

    selectEmploymentStatus = selectL = st.sidebar.selectbox("Select EmploymentStatus:",["Not_specified", "Fully", "Entrepreneur", "Retiree","Self_employed", "Partially", "unkown"])
    if (selectEmploymentStatus == "Not_specified"):
        selectEmploymentStatus = 0
    elif (selectEmploymentStatus == "Fully"):
        selectEmploymentStatus = 1
    elif (selectEmploymentStatus == "Entrepreneur"):
        selectEmploymentStatus = 2
    elif (selectEmploymentStatus == "Retiree"):
        selectEmploymentStatus = 3
    elif (selectEmploymentStatus == "Self_employed"):
        selectEmploymentStatus = 4
    elif (selectEmploymentStatus == "Partially"):
        selectEmploymentStatus = 5
    else:
        selectEmploymentStatus = 6

    selectEmploymentDurationCurrentEmployer = selectL = st.sidebar.selectbox("Select EmploymentDurationCurrentEmployer:",
                                                         ["MoreThan5Years", "UpTo1Year", "UpTo5Years", "UpTo2Years",
                                                          "UpTo3Years", "Retiree", "UpTo4Years","Other","unkown","TrialPeriod"])
    if (selectEmploymentDurationCurrentEmployer == "MoreThan5Years"):
        selectEmploymentDurationCurrentEmployer = 0
    elif (selectEmploymentDurationCurrentEmployer == "UpTo1Year"):
        selectEmploymentDurationCurrentEmployer = 1
    elif (selectEmploymentDurationCurrentEmployer == "UpTo5Years"):
        selectEmploymentDurationCurrentEmployer = 2
    elif (selectEmploymentDurationCurrentEmployer == "UpTo2Years"):
        selectEmploymentDurationCurrentEmployer = 3
    elif (selectEmploymentDurationCurrentEmployer == "UpTo3Years"):
        selectEmploymentDurationCurrentEmployer = 4
    elif (selectEmploymentDurationCurrentEmployer == "Retiree"):
        selectEmploymentDurationCurrentEmployer = 5
    elif (selectEmploymentDurationCurrentEmployer == "UpTo4Years"):
        selectEmploymentDurationCurrentEmployer = 6
    elif (selectEmploymentDurationCurrentEmployer == "Other"):
        selectEmploymentDurationCurrentEmployer = 7
    elif (selectEmploymentDurationCurrentEmployer == "unkown"):
        selectEmploymentDurationCurrentEmployer = 8
    else:
        selectEmploymentDurationCurrentEmployer = 9

    selectOccupationArea = selectL = st.sidebar.selectbox("Select OccupationArea:",
                                                     ["Not_specified",
                                                      "Other", "Retail_and_wholesale",
                                                      "Construction", "Processing", "Transport_and_warehousing", "Healthcare_and_social_help",
                                                      "Hospitality_and_catering", "Info_and_telecom", "Civil_service_and_military",
                                                      "Education", "Finance_and_insurance",
                                                      "Agriculture_forestry_and_fishing",
                                                      "Administrative",
                                                      "Energy", "Art_and_entertainment",
                                                      "Research","Real_estate","Utilities","Mining"])
    if (selectOccupationArea == "Not_specified"):
        selectOccupationArea = 0
    elif (selectOccupationArea == "Other"):
        selectOccupationArea = 1
    elif (selectOccupationArea == "Retail_and_wholesale"):
        selectOccupationArea = 2
    elif (selectOccupationArea == "Construction"):
        selectOccupationArea = 3
    elif (selectOccupationArea == "Processing"):
        selectOccupationArea = 4
    elif (selectOccupationArea == "Transport_and_warehousing"):
        selectOccupationArea = 5
    elif (selectOccupationArea == "Healthcare_and_social_help"):
        selectOccupationArea = 6
    elif (selectOccupationArea == "Hospitality_and_catering"):
        selectOccupationArea = 7
    elif (selectOccupationArea == "Info_and_telecom"):
        selectOccupationArea = 8
    elif (selectOccupationArea == "Civil_service_and_military"):
        selectOccupationArea = 9
    elif (selectOccupationArea == "Education"):
        selectOccupationArea = 10
    elif (selectOccupationArea == "Finance_and_insurance"):
        selectOccupationArea = 11
    elif (selectOccupationArea == "Agriculture_forestry_and_fishing"):
        selectOccupationArea = 12
    elif (selectOccupationArea == "Administrative"):
        selectOccupationArea = 13
    elif (selectOccupationArea == "Energy"):
        selectOccupationArea = 14
    elif (selectOccupationArea == "Art_and_entertainment"):
        selectOccupationArea = 15
    elif (selectOccupationArea == "Research"):
        selectOccupationArea = 16
    elif (selectOccupationArea == "Real_estate"):
        selectOccupationArea = 17
    elif (selectOccupationArea == "Utilities"):
        selectOccupationArea = 18
    elif (selectOccupationArea == "Mining"):
        selectOccupationArea = 19
    else:
        selectOccupationArea = 20

    selectHomeOwnershipType = selectL = st.sidebar.selectbox("Select HomeOwnershipType:",
                                                     ["Owner","Tenant_pre_furnished_property", "Living_with_parents",
                                                      "Mortgage", "Tenant_unfurnished_property", "Other",
                                                      "Joint_ownership", "unkown", "Joint_tenant", "Council_house",
                                                      "Owner_with_encumbrance",
                                                      "Homeless", "Not_specified"])
    if (selectHomeOwnershipType == "Owner"):
        selectHomeOwnershipType = 0
    elif (selectHomeOwnershipType == "Tenant_pre_furnished_property"):
        selectHomeOwnershipType = 1
    elif (selectHomeOwnershipType == "Living_with_parents"):
        selectHomeOwnershipType = 2
    elif (selectHomeOwnershipType == "Mortgage"):
        selectHomeOwnershipType = 3
    elif (selectHomeOwnershipType == "Tenant_unfurnished_property"):
        selectHomeOwnershipType = 4
    elif (selectHomeOwnershipType == "Other"):
        selectHomeOwnershipType = 5
    elif (selectHomeOwnershipType == "Joint_ownership"):
        selectHomeOwnershipType = 6
    elif (selectHomeOwnershipType == "unkown"):
        selectHomeOwnershipType = 7
    elif (selectHomeOwnershipType == "Joint_tenant"):
        selectHomeOwnershipType = 8
    elif (selectHomeOwnershipType == "Council_house"):
        selectHomeOwnershipType = 9
    elif (selectHomeOwnershipType == "Owner_with_encumbrance"):
        selectHomeOwnershipType = 10
    elif (selectHomeOwnershipType == "Homeless"):
        selectHomeOwnershipType = 11
    else:
        selectHomeOwnershipType = 12

    Rating = st.sidebar.selectbox("Enter the rating issued by the rating Model:",
                                  ["A", "AA", "B", "C", "D", "E", "F", "HR", "Empty"])
    if (Rating == "A"):
        Rating = 0
    elif (Rating == "AA"):
        Rating = 1
    elif (Rating == "B"):
        Rating = 2
    elif (Rating == "C"):
        Rating = 3
    elif (Rating == "D"):
        Rating = 4
    elif (Rating == "E"):
        Rating = 5
    elif (Rating == "Empty"):
        Rating = 6
    elif (Rating == "F"):
        Rating = 7
    elif (Rating == "HR"):
        Rating = 8
    else:
        Rating = 9

                        



    prevLoans=st.sidebar.number_input("Enter the value of : PreviousRepaymentsBeforeLoan")
    monthlyPayment=st.sidebar.number_input("Enter the value of:monthlyPayment")
    principalPayments=st.sidebar.number_input("Enter the value of:principalPayments")
    Mgr=st.sidebar.number_input("Enter the value of:BidsPortfolioManager")
    interest = st.sidebar.number_input("Enter the value of:interest")
    appliedAmount = st.sidebar.number_input("Enter the value of:appliedAmount")
    IAPP = st.sidebar.number_input("Enter the value of:InterestAndPenaltyPaymentsMade")
    BidsApi = st.sidebar.number_input("Enter the value of:BidsApi")
    BidsManual = st.sidebar.number_input("Enter the value of:BidsManual")
    Age = st.sidebar.number_input("Enter the value of:Age")
    IncomeTotal = st.sidebar.number_input("Enter the value of:IncomeTotal")
    ExistingLiabilities = st.sidebar.number_input("Enter the value of:ExistingLiabilities")
    RefinanceLiabilities = st.sidebar.number_input("Enter the value of:RefinanceLiabilities")
    DebtToIncome = st.sidebar.number_input("Enter the value of:DebtToIncome")
    FreeCash = st.sidebar.number_input("Enter the value ofe:FreeCash")
    CreditScoreEsMicroL = st.sidebar.number_input("Enter the value of:CreditScoreEsMicroL")

    data={"PreviousEarlyRepaymentsCountBeforeLoan":prevLoans,
          "MonthlyPayment":monthlyPayment,
          "PrincipalPaymentsMade":principalPayments,
          "BidsPortfolioManager":Mgr,
          "Interest":interest,
          "AppliedAmount":appliedAmount,
          "InterestAndPenaltyPaymentsMade" :IAPP,
          "BidsApi":BidsApi,
          "BidsManual" :BidsManual,
          "Age":Age,
          "IncomeTotal":IncomeTotal,
          "ExistingLiabilities":ExistingLiabilities,
          "RefinanceLiabilities":RefinanceLiabilities,
          "DebtToIncome":DebtToIncome,
          "FreeCash":FreeCash,
          "CreditScoreEsMicroL":CreditScoreEsMicroL,
          "LanguageCode": Language,
          "Country": selectCountry,
          "NewCreditCustomer": selectNewCreditCustomer,
          "Restructured": selectRestructured,
          "VerificationType": selectVerificationType,
          "Gender": selectGender,
          "UseOfLoan": selectUseOfLoan,
          "Rating": Rating,
          "Education": selectEducation,
          "MaritalStatus": selectMaritalStatus,
          "EmploymentStatus": selectEmploymentStatus,
          "EmploymentDurationCurrentEmployer": selectEmploymentDurationCurrentEmployer,
          "OccupationArea": selectOccupationArea,
          "HomeOwnershipType": selectHomeOwnershipType
          }

    features=pd.DataFrame(data, index=[0])
    return features

df=InputData()

st.subheader("Borrower Input Parameters:")
st.write(df)
clf=RandomForestClassifier()
linear = LinearRegression()
ridge = Ridge()




#def Modeling():
#
#    enc = OrdinalEncoder()
#    le = LabelEncoder()
#    def EncodingCategoricals(columns, df):
#        for c in columns:
#            df[c] = le.fit_transform(df[c])
#    objects = Data.select_dtypes("object")
#    EncodingCategoricals(Data.columns, Data)
#
#
#    y1=Data["Default"]
#    y2=Data[["EMI","ELA","ROI"]]
#    drop_target = ["Default","EMI","ELA","ROI"]
#    X=Data.drop(["Default","EMI","ELA","ROI"], axis=1)
#    sc=StandardScaler()
#    X=sc.fit_transform(X)
#    X_train, X_test, y_train, y_test = train_test_split(X, y1,test_size=0.25, random_state=0)
#    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.25, random_state=0)
##    transformer1 = ColumnTransformer([('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'),[3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17, 18, 24, 25])], remainder='passthrough')
##    transformer2 = ColumnTransformer([('scaling', StandardScaler(), [0, 1, 2, 6, 9, 10, 11, 19, 20, 21, 22, 23, 26, 27, 28, 33])],remainder='passthrough')
##    transformer3 = Ridge(alpha=2.0)
##    Pipeline = make_pipeline(transformer1, transformer2, transformer3)
##    estimators = [('svr', Pipeline)]
##    clf = StackingClassifier(estimators=estimators, final_estimator=Ridge())
##    Pipeline.fit(X_train, y_train)
#
##    self.estimators_ = [clone(self.base_estimator) for _ in range(y2.shape[1])]
#    clf.fit(X_train,y_train)
#    ridge.fit(X_train2, y_train2)
#
#
#st.subheader("Loan Status:")
#def PredictAns():
#    Modeling()
#    y_pred_ridge=ridge.predict(df)
#    y_predict=clf.predict(df)
# #   y_pred_pip = Pipeline.predict(df)
#
#    return y_predict,y_pred_ridge
#
#if(st.sidebar.button("Predict")):
#    ans,ans3 =PredictAns()
#    outputData = {"EMI": ans3[0][0],
#                  "ELA": ans3[0][1],
#                  "ROI": ans3[0][2]
#                  }
##    st.write(ans2)
##    st.write(ans3)
#
#    if(ans==0):
#        st.write("Congratulations! The selected loan has been admitted :)")
##        st.write(ans2)
##        st.write(ans2)
#
#        st.write(outputData)
#    else:
#        st.write("Unfortunately, The selected loan has been denied :(")
##        st.write(ans2)
##        st.write(ans2)
#
#        st.write(outputData)
#///////////////////////////////////////////////////////////////////////
cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
enc = OrdinalEncoder()
X_cat_enc = enc.fit_transform(df[cat_cols])
df[cat_cols] = X_cat_enc


clf=RandomForestClassifier()
model = Pipeline([('Scaler', StandardScaler()), ('clf', RandomForestClassifier())])

def clfModel():
    cat_cols = Data.select_dtypes(exclude=['int64', 'float64']).columns
    num_cols = Data.select_dtypes(include=['int64', 'float64']).columns
    X = Data.drop(['Default', 'EMI', 'ELA', 'ROI'], axis=1)
    y = Data['Default']
    enc = OrdinalEncoder()
    X_cat_enc = enc.fit_transform(X[cat_cols])
    X[cat_cols] = X_cat_enc
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    clf.fit(X_train,y_train)


st.subheader("The result:")


def Prediction():
    clfModel()
    y_predict = clf.predict(df)
    return y_predict


if (st.sidebar.button("Predict")):
    val = Prediction()
    if (val == 0):
        st.write("Congratulations! The loan has been admitted :)")
    else:
        st.write("Unfortunately, The loan has been rejected :(")

    cat_cols = Data.select_dtypes(exclude=['int64', 'float64']).columns
    num_cols = Data.select_dtypes(include=['int64', 'float64']).columns
    X = Data.drop(['Default', 'EMI', 'ELA', 'ROI'], axis=1)
    y = Data[['EMI', 'ELA', 'ROI']]
    enc = OrdinalEncoder()
    X_cat_enc = enc.fit_transform(X[cat_cols])
    X[cat_cols] = X_cat_enc

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .75)

    transformer2 = ColumnTransformer([('scaling', StandardScaler(), [0, 1, 2, 6, 9, 10, 11, 19, 20, 21, 22, 23, 26, 27, 28,29])],
                                     remainder='passthrough')
    transformer3 = Ridge(alpha=2.0)
    Pipeline1 = make_pipeline(transformer2, transformer3)
    Pipeline1.fit(X_train, y_train)
    st.subheader("The Credit Risk:")
    outputData = {"EMI": Pipeline1.predict(df)[0][0],"ELA": Pipeline1.predict(df)[0][1],"ROI": Pipeline1.predict(df)[0][2]}
    st.write(outputData)
#    st.subheader("EMI:")
#    st.write(Pipeline1.predict(df)[0][0])
#    st.subheader("ELA:")
#    st.write(Pipeline1.predict(df)[0][1])
#    st.subheader("ROI:")
#    st.write(Pipeline1.predict(df)[0][2])

















#    transformer1 = ColumnTransformer([('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'),[3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 17, 18, 24, 25])], remainder='passthrough')
#    transformer2 = ColumnTransformer([('scaling', StandardScaler(), [0, 1, 2, 6, 9, 10, 11, 19, 20, 21, 22, 23, 26, 27, 28, 33])],remainder='passthrough')
#    transformer3 = Ridge(alpha=2.0)
#    Pipeline = make_pipeline(transformer1, transformer2, transformer3)
#    Pipeline.fit(X_train, y_train)

#    y_pred_pip = Pipeline.predict(df)



#    X_cat_enc = enc.fit_transform(X[cat_cols])
#   X[cat_cols] = X_cat_enc

#X = Data.copy()
#y = X.pop('Default')
#RandemForest(X)


#s@t.cache(persist= True)
#def load():
#    data= pd.read_csv("finalData2.csv")
#    label= LabelEncoder()
#    for i in data.columns:
#        data[i] = label.fit_transform(data[i])
#    return data
#    df = load()
#
#if st.sidebar.checkbox("Display data", False):
#    st.subheader("Show Mushroom dataset")
#    st.write(df)
#
#@st.cache(persist=True)
#def split(df):
#    y = df.Default
#    X = df.drop(columns=["Default"])
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
#
#    return X_train, X_test, y_train, y_test
#    X_train, X_test, y_train, y_test = split(df)
#
#
#
#def plot_metrics(metrics_list):
#    if "Confusion Matrix" in metrics_list:
#        st.subheader("Confusion Matrix")
#        plot_confusion_matrix(model, X_test, y_test, display_labels=   class_names)
#        st.pyplot()
#    if "ROC Curve" in metrics_list:
#        st.subheader("ROC Curve")
#        plot_roc_curve(model, X_test, y_test)
#        st.pyplot()
#    if "Precision-Recall Curve" in metrics_list:
#        st.subheader("Precision-Recall Curve")
#        plot_precision_recall_curve(model, X_test, y_test)
#        st.pyplot()
#    class_names = ["edible", "poisnous"]
#
#
#st.sidebar.subheader("Choose classifier")
#classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression", "Random Forest"))
#
#if classifier == "Logistic Regression":
#    st.sidebar.subheader("Hyperparameters")
#    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
#    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
#    metrics = st.sidebar.multiselect("What metrics to plot?",
#                                     ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
#
#    if st.sidebar.button("Classify", key="classify"):
#        st.subheader("Logistic Regression Results")
#        model = LogisticRegression(C=C, max_iter=max_iter)
#        model.fit(X_train, y_train)
#        accuracy = model.score(X_test, y_test)
#        y_pred = model.predict(X_test)
#
#        st.write("Accuracy: ", accuracy.round(2))
#        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
#        plot_metrics(metrics)
#
#if classifier == "Random Forest":
#    st.sidebar.subheader("Hyperparameters")
#    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
#    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step=1, key="max_depth")
#    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
#
#    metrics = st.sidebar.multiselect("What metrics to plot?",
#                                     ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
#
#    if st.sidebar.button("Classify", key="classify"):
#        st.subheader("Random Forest Results")
#        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
#        model.fit(X_train, y_train)
#        accuracy = model.score(X_test, y_test)
#        y_pred = model.predict(X_test)
#
#        st.write("Accuracy: ", accuracy.round(2))
#        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
#        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
#        plot_metrics(metrics)
       # One hot encoding for categorical variables
#cats = Data.dtypes == 'object'
#le = LabelEncoder()
#for x in Data.columns[cats]:
#    Data.loc[:, x] = le.fit_transform(Data[x])
#    onehotencoder = OneHotEncoder()
#    Data.loc[:, ~cats].join(pd.DataFrame(Data=onehotencoder.fit_transform(Data.loc[:, cats]).toarray(), columns=onehotencoder.get_feature_names()))

#def RandemForest(X):
#    clf = RandomForestClassifier()
#
#    sc = StandardScaler()
#    X = sc.fit_transform(X)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)