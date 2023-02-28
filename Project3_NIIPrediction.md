# Project 3 - Net interest income prediction¶
- Description: Base on the Bank's nature, it's history performance and the previous year Balance Sheet, this year planning disbursement (in this project, ENR is used instead), this project will provide the estimation of Net interest income (NII) of the year.
- Project's value: This project helps BOM in planning the Targetted Disbursement volumes (here ENR instead) of the year so as to achieve the NII, as well as Profit
- Data source: Public Financial Statements of the banks

## 1. Data loading and droping missing value
    import pandas as pd

    df = pd.read_csv('Raw.csv',sep= ',').dropna()
    df.columns = df.columns.str.strip()
    df
![image](https://user-images.githubusercontent.com/89530538/221657122-860b57c7-d29f-4757-8b3d-e6c87a0194e5.png)

    
    # Xử lý khoản trắng và các ký tự đặc biệt
    df.columns = df.columns.str.strip()
    df = df.replace({'\$': ''}, regex=True)
    df
![image](https://user-images.githubusercontent.com/89530538/221657417-fde70801-dbc4-4542-877b-3220b7f4e83f.png)

    # Chuyển đổi string thành dạng float
    cols = ['Cur_TA_ENR', 'Pre_Total_Asset', 'Pre_TA_ENR', 'Pre_Liability', 'Pre_Equity', 'Pre_Retain_earning', 'NII']
    df[cols] = df[cols].astype(float)
    
## 2. Splitting dataset

    y = df['NII']
    x = df.drop(['NII','Bank'], axis=1)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train , y_test = train_test_split(x,y, train_size=0.8, random_state=0)

## 3. Training Model

    from sklearn.tree import DecisionTreeRegressor
    dt_model = DecisionTreeRegressor(random_state=1)

    dt_model.fit(x_train,y_train)
    y_train_pre = dt_model.predict(x_train)
    y_test_pre = dt_model.predict(x_test)
    
## 4. Model Evaluation

    from sklearn.metrics import mean_squared_error, r2_score

    lr_train_mse = mean_squared_error(y_train, y_train_pre)
    lr_train_r2 = r2_score(y_train, y_train_pre)

    lr_test_mse = mean_squared_error(y_test, y_test_pre)
    lr_test_r2 = r2_score(y_test, y_test_pre)
