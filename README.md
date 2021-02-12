


# 2021MCM _ Problem_Code

----------
## Environmental Dependency:python3.6 pytorch cudatoolkit plotly sklearn statemodels

----------

## 目录结构描述
    2021MCM _ Problem_Code
      ----Problem1
          ----GaussianMixture.py
      ----Problem2
          ----config.py
          ----dataprocessing.py
          ----Mydataloader.py
          ----myimportlib.py
          ----second_model.py
          ----test.py
          ----train.py
      ----Problem3
          ----myanova.py
      ----Problem4
          ----pred.py
      ----Problem5
          ----config.py
          ----dataprocessing.py
          ----getmodule.py
          ----Model.py
      ----Others
          ----data
              ---file
          ----filterdata.py
          ----filterdata2.py

----------
##########How To Use
- **Others/data contains all data files that have been cleaned and not cleaned**
- **Others/filterdata.py and Others/filterdata2.py use for data pre-cleaning**
- **Problem1/GaussianMixture/train_timepred use to solve problems,Problem1/GaussianMixture/Paint and Paint2 and Paint3 use plotly to implement data visualization**
- **Problem2/train.py Problem2/test.py implement model training and testing**
- **Problem3 use myanova(pd_dataprocessing()) to implement variance analysis**
- **Problem4 use SVR_logistic2("../Others/data/Five_data.csv",5) to implement time series and solve problem4**
- **Problem5/dataprocessing/Train_solve("../Others/data/Five_data.csv", 10, 10) implement model training and testing**




