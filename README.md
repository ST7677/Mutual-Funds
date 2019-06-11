# Mutual-Funds

The objective of this project is to extract latest data on Indian Mutual Funds from public sources for analysis and recommend top 5 funds as good investment options in different categories for further detailed analysis. As part of this I  also explore how different factors influence the performance for different investment horizons. 

## Motivation

Mutual fund are professionally managed investment fund that pools money from many investors to purchase securities. These investors may be retail or institutional in nature.
Mutual funds have advantages and disadvantages compared to direct investing in individual securities. The primary advantages of mutual funds are that they provide economies of scale, a higher level of diversification, they provide liquidity, and they are managed by professional investors. On the negative side, investors in a mutual fund must pay various fees and expenses.

This project was taken up as part of UDACITY Data Science Nano Degree program Capstone project. 


## Detailed Blog
There is a detailed blog with a walk through of the project available at - https://medium.com/@sunilthakur_67045/which-mutual-fund-87935649f1d5 


## Source Code and Data File
The Source Code for the project is available as MutualFunds.ipynb in this Git. The data is collected at run time from MoneyControl.com.
To Run this project you would need Python 3.xx 

## Python Libraries used
numpy,
pandas,
matplotlib,
seaborn,
sklearn,
RandomForestClassifier from sklearn.ensemble,
csv,
BeautifulSoup from bs4,
requests,
sys,
re

## Results Analysis
I generated feature importance charts from random forest classification to visualize about the features that are important predictors in each of the time horizons, and as expected CRISIL rating is one of the most dominating features across different time horizons.

I achieved following classification accuracy scores for different time horizons
1 month : 93%, 
3 months : 93%, 
6 months : 94%, 
1 year : 94%, 
2 year : 94%, 
3 year : 93%, 
5 year : 93%

But this is primarily due to class imbalance problem, i.e. if the model marks everything as 0 (not good fund) then also it will get very good accuracy. I can observe this by calculating precision and recall and plotting RoC curve (Receiver Operating Characteristics). This problem may be solved by either increasing number of good samples (by duplicating the good fund records or by reducing the number of bad funds (not recommended here).

Finally the model recommends a shortlist of top 5 funds across different time horizons, for user to analyze the same in detail for investment.

## Future Improvements
In future I would like to add various risk and volatility features (Sharpe Ratio, Sortino Ratio, Standard Deviation, R-Squared, Beta, Alpha, Treynor Ratio, etc) to incorporate modern portfolio statistics in this model. I will need to search for this information online for all the funds and merge the data pipeline from two different websites to achieve this. I would also like to address the class imbalance problem in future.

## References and Acknowledgement

http://www.moneycontrol.com/mutualfundindia/ - for making data available

http://udacity.com/ - For providing an excellent course in Data Sceince and their thanks to Udacity mentors for providing helpful feedback. 
