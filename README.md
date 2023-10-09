# Ransomware Bitcoin Transaction Analysis

## Abstract
Ransomware attacks have emerged as a significant concern in the realm of cybercrime, with Bitcoin transactions frequently employed for ransom demands. Identifying and locating such transactions is critical for addressing this problem. In this project, we propose the use of machine learning models to detect ransom-related Bitcoin transactions and categorize them by their location. The dataset spans from 2009 to 2018 and comprises over 2.9 million instances with 11 attributes.

To handle this extensive dataset, we use big data technologies like Spark for parallel processing and faster outcomes. We assess the efficacy of various machine learning models, including logistic regression, an ensemble of logistic regression, a decision tree, and a random forest, for detecting ransomware-associated transactions. With the use of bagging, we combine the results of multiple models and evaluate their performance using metrics such as recall, precision, and the macro F1 score

Our research highlights the value of incorporating big data technologies and multiple machine learning models in detecting ransom-related transactions. The decision tree model outperformed others in terms of accuracy and macro F1 score, demonstrating its effectiveness in differentiating between ransom-related and genuine transactions. This study offers insights into the efficiency of different machine learning models for identifying ransom-associated Bitcoin transactions and their locations, emphasizing the advantages of utilizing big data technologies in this context.

## Table of Contents
1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Datasets](#datasets)
4. [Methodology](#methodology)
5. [Experimental Set-up](#experimental-set-up)
6. [Results and Discussion](#results-and-discussion)
    - [Performance Comparison of Models](#performance-comparison-of-models)
    - [Poor Performance of LR and ELR Models](#poor-performance-of-lr-and-elr-models)
    - [Advantages of Tree-Based Models](#advantages-of-tree-based-models)
    - [Decision Tree Model Advantages](#decision-tree-model-advantages)

## Introduction
Blockchain is a distributed public ledger used for recording transactions between two parties. It consists of a chain of blocks, each containing permanent transaction records. Since it's a public ledger, it's transparent and open for public auditing. To ensure security, cryptography methods are used to encrypt transactions, allowing only the parties involved to decrypt and view them. Blockchain technology was invented in 2008, with Bitcoin being one of the first applications to utilize it. The use of Bitcoin has grown rapidly in recent years due to its ability to facilitate global transactions without exchange fees and without being controlled by any organization or government.

Bitcoin enables peer-to-peer transactions without a central authority. Transactions are anonymous, and no identity verification is needed to use Bitcoin. With just a public address, senders can easily complete transactions. The global and pseudonymous nature of Bitcoin transactions has been exploited by cybercriminals to commit cybercrimes, particularly ransomware attacks.

Ransomware, malicious software that locks files on a victim's device, has significantly increased in the realm of cybercrime by taking advantage of Bitcoin transactions. Attackers send ransomware to individuals, businesses, or governments to encrypt their files and demand payment to a Bitcoin address to obtain the decryption key. The impact of ransomware extends beyond victims, affecting the economy and society. Therefore, it is crucial to detect ransom payments among Bitcoin transactions to prevent attackers from continuing to spread crypto-ransomware.

This project aims to implement a machine-learning solution with a big data approach to identify ransom-related Bitcoin transactions and their originating cities based on the transaction graph. The contributions of this project include developing a novel approach for detecting ransom payments within the vast network of Bitcoin transactions and potentially aiding law enforcement agencies in combating cybercrime.

## Related Work
Ransomware attacks have become an increasingly significant issue in cybersecurity. The emergence of cryptocurrencies, particularly Bitcoin, has facilitated extortion by enabling anonymous transactions. To effectively combat this threat, it is critical to understand the underlying payment patterns and develop robust detection mechanisms.

Several studies have focused on analyzing Bitcoin transactions to identify ransomware patterns. Ron and Shamir were among the first to examine the Bitcoin transaction graph, highlighting the flow of funds from victims to attackers. Other researchers further explored the distinct characteristics of ransom payments and their movement within the Bitcoin network, uncovering valuable information about attacker behavior.

Machine learning techniques have proven useful in detecting illicit transactions related to cybercrime. Researchers have investigated shared patterns and heuristics in hacker behavior, which informed the development of algorithms for detecting ransom payments. Rule-based approaches and machine learning classifiers have been employed to detect money laundering activities within the Bitcoin ecosystem.

This section aims to link our submission to existing knowledge by providing a comprehensive overview of the literature on ransomware payments and their detection. By building upon the findings of these studies, our work contributes to the ongoing refinement and enhancement of methods for identifying and mitigating ransomware threats.

## Datasets
The dataset used in this study is obtained from UCI and comprises multivariate, time-series data of Bitcoin transaction graphs from 2009 to 2018. With over 2.9 million instances and 10 attributes, it encompasses a range of integer and real characteristics. The dataset was donated on June 17, 2020, and has been employed for classification and clustering tasks.

The rapid growth and increasing complexity of ransomware attacks have necessitated the use of big data techniques to better understand and mitigate these threats. By employing a comprehensive dataset of Bitcoin transactions, this study aims to leverage the power of big data analytics to identify patterns and trends associated with ransomware activities

The rationale behind using big data techniques in this context includes the ability to process and analyze large amounts of information quickly and accurately, as well as the opportunity to uncover hidden patterns and trends that may be difficult to detect using traditional analysis methods. By employing big data analytics, we can better discern the unique characteristics of ransomware transactions and develop more effective strategies for combating these malicious activities.

Each transaction in the dataset is labeled as either a genuine transaction or as a payment associated with one of 27 different types of ransomware. For the purposes of this study, ransomware labels were modified to indicate only the city of the attack. Consequently, our predictions focus on determining whether a given address corresponds to a genuine transaction or a ransom payment in Princeton, Padua, or Montreal.

## Methodology
The methodology used in this paper follows the flow depicted in Figure 5. The various steps of the methodology are explained below:

### A. Outlier Removal
As shown in Figure 1, the dataset contains outliers. Outliers in the dataset can significantly affect the accuracy and robustness of the predictions.

![image](https://github.com/Monish24/Ransom-Payment-Detection-/assets/54630644/2e019277-91f5-4601-bdee-d917d97f9bd0)


The Z-score standardizes the dataset by centering the mean and removing samples that were n steps ahead or behind. We set n to 3, which means that samples 3 steps away from the mean are removed. The use of the Z-score technique ensures that the dataset used for the analysis was accurate and reliable.

### B. Train-Test Split
A train-test split is performed to prevent information leakage when applying oversampling techniques during the training phase.

### C. Label Encoding
There are four types of ransomware families that we are interested in, each labeled by its family name. We performed label encoding to encode the label with 0 for Princeton, 1 for Padua, 2 for Montreal and 3 for White. This technique helps enable the analysis of the impact of ransomware family names on the prediction of ransom payments.

### D. Handling Imbalanced Data
The dataset contains 2,916,697 samples, with 98.6% labeled as "White" and only 1.4% associated with ransomware. This imbalance causes the models to be biased towards genuine transactions and results in poor ransom payment predictions.

![image](https://github.com/Monish24/Ransom-Payment-Detection-/assets/54630644/7e888340-723f-4b4c-95bb-4d8b99ffc317)

To mitigate the impact of imbalanced data, we employed the SMOTE-ENN technique, which performs both oversampling and undersampling. SMOTE generates synthetic samples for the minority class by interpolating between existing minority class samples.

![image](https://github.com/Monish24/Ransom-Payment-Detection-/assets/54630644/f7a29a4c-6ad0-4ddc-a62c-17ebed1769c3)

The Edited Nearest Neighbor (ENN) technique is an undersampling method that removes misclassified samples from the majority class based on the k-nearest neighbor classifier. In this project, we used Smote to generate synthetic data for the minority class to match the number of samples in the majority class and then applied ENN to remove noisy samples from the data. This approach, known as SMOTE-ENN, was only applied to the training subset to prevent data leakage in the test set.

### E. Normalizing Data
Normalizing the features scales their values into a specific range, which enhances model stability, especially for features with large value ranges such as "income". This process ensures that all features are on the same scale and have the same magnitude. We used a standard scaler to normalize the features, which scales them to unit variance.

### F. Logistic Regression (LR)
LR is based on the concept of linear regression. It uses a sigmoid function to map the output of linear regression to a value between 0 and 1, which represents the probability of belonging to a particular class. Specifically, the output of LR is either 1 or 0, indicating the likelihood of belonging to one class. LR was chosen for this study because of its simplicity and interpretability; it helped us analyze the impact of various features on the prediction of ransom payments.

### G. Decision Tree (DT)
Decision Trees learn to form a tree representation of the data, with leaf nodes representing outcomes and decision nodes containing conditions for data splitting. The Attribute Selection Method (ASM) is an intrinsic feature selection technique in DTs that helps determine optimal splitting conditions.

### H. Ensemble Method
Both logistic regression and decision tree were implemented into their own ensemble method using the bagging approach. Bagging is a popular ensemble learning technique that trains multiple classifiers with different random subsets of data and employs majority voting to obtain the final prediction.

Ensemble learning helps to reduce overfitting by combining the predictions of multiple models, each trained on a different subset of the data. The bagging approach is particularly effective because it randomly selects a subset of data to train each model, allowing it to learn from a more diverse set of samples. The exposure to a more diverse set of samples helps to reduce variance in the model predictions and thus makes a more robust and accurate prediction.

![image](https://github.com/Monish24/Ransom-Payment-Detection-/assets/54630644/d6ed3070-ff25-4234-881f-8f001cb97a2a)

As illustrated in Figure 4, the data selected for training is done with replacement, which means that some points may appear more than once in the training subsets. However, since the size of the resampled data is the same as the original dataset, some data may not be selected during the resampling process.

The ensemble approach for the decision tree model is called the random forest (RF) model, which is available in MLlib. The random forest model works by constructing multiple decision trees on different subsets of the training data and then combining their predictions through majority voting.

For ensemble logistic regression (ELR), the pseudocode for the ensemble approach is shown in Figure 4. The pseudocode outlines the steps involved in training multiple logistic regression models on different subsets of the training data and then combining their predictions to obtain the final prediction.

![image](https://github.com/Monish24/Ransom-Payment-Detection-/assets/54630644/c2ed0d0a-6642-4584-afd5-9fc908339ded)

## Experimental Set-up
### A. Data Pre-processing
To prepare the Bitcoin transaction dataset for analysis, several preprocessing steps were performed. First, the dataset was loaded using the spark-CSV library. The ransomware labels were then clustered by location, reducing the number of distinct labels and improving the interpretability of the models. Outliers were removed using z-mean by computing the means and standard deviations of each column and filtering out any rows where the absolute deviation from the mean was greater than 3 times the standard deviation. The input features were then scaled using a StandardScaler. The data was split into training and testing sets using a 70/30 split, and the training set was balanced using SMOTE-ENN to address the data imbalance.

### B. Hyperparameter Tuning
To optimize the performance of the models, hyperparameters were tuned using 5-fold cross-validation. For the Logistic Regression model, a LogisticRegression object was created with elasticNetParam and regParam hyperparameters. The ElasticNetParam hyperparameter grid was set to [0.0, 0.6, 1.0], and the RegParam hyperparameter grid was set to [0.01, 0.1, 1.0]. For the Random Forest model, a RandomForestClassifier object was created with numTrees and maxDepth hyperparameters. The numTrees hyperparameter grid was set to [15, 17, 19], and the maxDepth hyperparameter grid was set to [12, 15, 17]. For the Decision Tree model, a DecisionTreeClassifier object was created with maxDepth and maxBins hyperparameters. The maxDepth hyperparameter grid was set to an array range from 1 to 30, and the maxBins hyperparameter grid was set to [32, 64, 128]. The hyperparameters were tuned on k-fold cross-validation with k=5 and evaluator=MulticlassClassificationEvaluator(metricName="f1").

### C. Evaluation Metric
Relying solely on accuracy for model evaluation can be insufficient, as it might not show a model's true performance. To address this, the F1-score, which considers precision and recall, is used alongside accuracy. This enables a comprehensive assessment of the model's ability to predict positive and negative cases accurately. For multiclass problems, the macro F1-score, which averages the F1-score across all labels, is appropriate. A high macro F1-score indicates the model performs well on all labels. In summary, combining macro F1-score and accuracy provides a robust evaluation of the model's performance.

## Results and Discussion
### Performance Comparison of Models
Based on Table 6, the Decision Tree (DT) model outperforms other models in terms of accuracy, recall, and macro F1 score. The performance of the Random Forest (RF) model matches that of DT, which is expected since both are tree-based models. In contrast, the Logistic Regression (LR) and Extended Logistic Regression (ELR) models exhibit similar performance, but both perform poorly. Interestingly, the ensemble methods do not significantly outperform the single models. This suggests that complex models were not required for this problem, as the input features demonstrated a simpler relationship. As a result, simpler models were able to learn the pattern and make accurate predictions on the test dataset while saving computational costs and time.

### Poor Performance of LR and ELR Models
The LR model assumes a linear relationship between variables. If this assumption is not met, the model cannot fit well, resulting in poor performance. Thus, the non-linear relationship present in this dataset may be the primary cause of the poor performance. Since ELR uses LR as its basis, it also suffers from poor performance.

Another possible reason for the poor performance of LR and ELR models is the imbalanced dataset. Although the Smote-Enn technique is applied, the Edited Nearest Neighbors (Enn) process removes many white samples due to their lack of clustering, making the white label the minority. Table 2 and Table 4 show the F1 score, recall, and precision of LR and ELR. Both models have high precision but poor recall on the white label, leading to a low F1 score. This indicates that the models struggle to predict genuine transactions. Moreover, the results also suggest that these two approaches cannot accurately identify the location of ransomware attacks, as the F1 scores for other classes are very low. Overall, the logistic regression models fail to learn the pattern of the dataset effectively.

### Advantages of Tree-Based Models
Tree-based methods, such as DT and RF, have the advantage of learning non-linear relationships between variables. The performance of both tree-based models in Table 6 supports the presence of a non-linear relationship between variables. One advantage of using RF instead of DT is its lower susceptibility to overfitting. However, in this case, DT slightly outperforms RF, likely because DT is not overfitted to the training data. Consequently, RF does not offer a significant improvement in performance.

Additionally, tree-based methods exhibit a slight advantage in handling imbalanced data after applying SMOTE-ENN. Table 3 and Table 5 present the results for DT and RF in terms of precision, recall, and F1 score for each label. Notably, both models achieve high F1 scores on the white label, indicating their ability to classify ransom-related transactions and genuine transactions. However, these models struggle to group ransom-related transactions by location, resulting in high recall but low precision for three labels. This suggests that there may not be significant differences in ransom-related payments across different cities, making it difficult for the model to identify ransom payments by location. Overall, the tree-based models can effectively differentiate between normal Bitcoin transactions and ransom-related transactions but struggle to predict the location of ransomware attacks.

### Decision Tree Model Advantages
There are several benefits to using the DT model over the RF model. First, DT is a highly interpretable model that allows for the identification of important blockchain graph features in detecting ransom payments. Additionally, DT is a lightweight algorithm capable of making predictions quickly. Although data scaling was performed in this study, DT can also be trained on unscaled data, further reducing prediction time when implemented in the Bitcoin system. This improves the model's scalability, as the rapidly growing number of Bitcoin transactions requires quick decision-making. Consequently, the decision tree model emerges as the best-performing model based on our experiment.

## Conclusion

![image](https://github.com/Monish24/Ransom-Payment-Detection-/assets/54630644/5655ee3b-abd9-4f08-9526-ae40dcf6e2e3)

Our results demonstrate that tree-based models, particularly the Decision Tree model, outperform other models in terms of accuracy and macro F1 score. These models can effectively differentiate between ransom-related transactions and genuine transactions but struggle to predict the location of ransomware attacks accurately.

Our research highlights the importance of considering both big data technologies and appropriate machine learning models in the detection of ransom-related Bitcoin transactions. While complex models like ensemble methods did not significantly outperform simpler models in this context, the use of big data technologies for processing and analyzing vast datasets proved valuable.

Future work in this area may involve exploring more advanced machine learning techniques, improving feature engineering, and incorporating additional data sources to enhance the accuracy of ransomware detection and location prediction in Bitcoin transactions.
