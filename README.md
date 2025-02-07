# Blueberry-Winery-Project ML Classification
White and Red Wines quality analysis with ML classification 
<p><b>TECHNOLOGIES USED:</b></p>
<p> import numpy as np </p>
<p> import pandas as pd </p>
<p> import matplotlib.pyplot as plt </p>
<p> import matplotlib.patches as mpatches </p>  
<p> import seaborn as sns </p>
<p> from scipy import stats </p>
<p> from sklearn.linear_model import LogisticRegression </p>
<p> from sklearn.preprocessing import MinMaxScaler </p>
<p> from sklearn.metrics import classification_report </p>
<p> from sklearn.metrics import confusion_matrix </p>
<p> from sklearn.model_selection import train_test_split </p>
<p> from sklearn.neighbors import KNeighborsClassifier </p>
<p> from sklearn.metrics import accuracy_score, confusion_matrix, classification_report </p>
<p> from sklearn.ensemble import RandomForestClassifier </p>
<p> from sklearn.model_selection import StratifiedKFold, cross_val_score </p>
<p> from sklearn.model_selection import GridSearchCV, StratifiedKFold </p>

**Data from Vinho Verde**
I received 2 dataset of white wine and red wine samples from Vinho Verde
The white wine dataset contains 4898 records of white wine samples which is 75% of the analysis research.
The red wine dataset contains 1599 records of red wine samples which is 25% of the analysis research.

**Vinho Verde**
Vinho verde refers to Portuguese wine that originated in the Minho province far north the country, due to its proximity to the Atlantic Ocean, abundancy in water bodies like rivers and streams, as well as the mild climate and high rainfall. The Vinho Verde Wine Region is gifted in terms of water supplies, turning it into a huge green landscape. From forests to fields, the region is very fertile, and green is its main hue.
Vinho Verde is not a grape variety, it is a DOC (Controlled Designation of Origin) for the production of wine, the name means “young wine”.

**Vinho Verde’s White wine and Red wine**
The Majority of the wines classified as Vinho Verde are white. The white Vinho verde is very fresh due to its natural acidity and are Lemon or straw-coloured, around 8,5 - 11% alcohol, most are a touch fizzy, mostly dry, and have green fruit notes. Tasting Notes: Lemonade, White Melon, Gooseberry, Grapefruit, and Lime Blossom.
The red and rosé Vinho Verde are much less common than the white ones, that is caused by the Region’s climatic condition with its relatively cool temperatures and high level of rainfall that make it impossible for the red wine grapes to ripen, So if you get your hands on a bottle – no matter the price, – you are drinking rare juice! Tasting Notes: Sour Plum, Sour Cherry, Pepper, and Peony.

**Wine Properties**
Fixed Acidity: acids are major wine properties and contribute greatly to a wine’s taste and is divided into two groups:
Volatile acidity: is basically the process of wine turning into vinegar. Amount should vary between 0,72g/l - 90g/l, higher amount are considered undesirable or unpleasant
Citric acid: is often added to wine to increase acidity, complement a specific flavour or prevent ferric hazes. 
Residual Sugar: is the remaining sugar after the fermentation and varies from 3g/l up to 50g/l -150g/l
Chlorides: is a significant contributor to saltiness in wine, and of the overall taste and quality of the wine
Free / Total sulfur dioxide: Winemakers try to have the highest proportion of  sulfur dioxide to bind. If is too low the wine is at risk for developing microbial axidative faults
Density: Is generally used as a measure of the conversion of sugar to alcohol.
pH: Specifies the acidity or basicity of the wine, most wines have a pH between 2.9 and 3.9, are therefore acidic
Sulfates: Are a regular part of wine making and considered necessary, but also can cause headaches.
Alcohol: Wine is an alcoholic beverage and the percentage of alcohol can vary from wine to wine

**General overview of both wine properties**
As we can notice, 8 chemical components in both wines are very different, therefore in order to Get accurate results we will analyse the two separately

**final thoughts on red wine**
Key Points for High-Quality Red Wine
Balance of Sulphates and Sulfur Dioxide: In red wine, achieving low sulfur dioxide levels while maintaining stability can contribute to a purer taste profile, enhancing quality.
Density and Alcohol Content: Higher alcohol with a corresponding denser body can contribute positively to mouthfeel and depth, which is typically desired in high-quality reds.
pH and Acidity Balance: In red wines, keeping a lower pH with adequate citric acid while controlling volatile acidity can preserve freshness and reduce spoilage risks, which aligns with high-quality characteristics 

**final thoughts on white wine**
Putting It All Together for High-Quality White Wine 
Balanced Sulfur Dioxide: Finding the right sulfur dioxide level is key. Too much can affect flavor, but too little can leave wine susceptible to spoilage. Lower sulfur dioxide can also help achieve a more natural flavor profile, which often appeals to consumers of high-quality wines.
Alcohol and Density Balance: Higher alcohol content can add body and warmth to the wine, which is often valued in quality wines. However, this should be balanced, as overly high alcohol can dominate other flavors.
pH Balance and Acidity: The ideal pH for white wines usually falls between 3.0 and 3.3. A balanced acidity level, with moderate citric acid, will maintain this ideal pH range, enhancing the wine’s freshness and overall stability.

**Machine learning**
I developed a machine learning model to predict the quality of white and red wine. To identify the most accurate approach, I tested three different algorithms: Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN). Given the relatively small number of high-quality wine samples, I adjusted the classification slightly, reclassifying some of the higher-medium quality wines as high quality to ensure the models could make meaningful predictions.
After extensive testing across these models, the Random Forest algorithm delivered the best results, achieving an accuracy of 73% for red wine and 72% for white wine.


**ML results for Red wine quality prediction**
For low quality red wine we can observe that the model predicted that 36.56% are low quality wine, 6.88 are mid quality and 0.62% is hight quality wine (both false)
For Mid quality, it predicted 9.38% is high quality (which is not true), 27.81% is mid quality (which is right) and 4.06% hight quality (which is false)
For high quality wine it predicted that 0.62% is low quality (false), 5.31% is mid quality (which is false) and high quality wine 8.75% which is true
We can observe that the data is evenly distributed looking at the heat map diagonally.

**ML results for White wine quality prediction**
For low quality red wine we can observe that the model predicted that 18.06% are low quality wine (which is true) 14.18 are mid quality and 0.51% is hight quality wine (both false)
For Mid quality, it predicted 9.90% is high quality (which is not true), 29.49% is mid quality (which is right) and 4.69% hight quality (which is false)
For high quality wine it predicted that 1.73% is low quality (false), 14.18% is mid quality (which is false) and high quality wine 7.24% which is true
We can observe that the data is not evenly distributed looking at the heat map diagonally, the machine struggled to detect the high quality wine because of maybe some data leakage somewhere. It needs to be fixed!
