# Amazon Product Reviews Sentiment Analysis


## Introduction

In trying to understand online customer behaviour and sentiments we were tasked with analysing the data from e-retail juggernaut Amazon. The company wants to be at the forefront in the beauty and personal care retail industry  which has been rapidly growing over the years with its market value expected to reach 558 Billion USD by 2026 according to the research done by The Business Wire. The company’s marketing department has thus sought to contract our services as data scientists to provide insights that will enable them capitalize on acquiring a vast portion of this forecasted market value. We will be advising the various brands that get good/neutral/bad reviews on how to improve their product specs. 



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
      <ul>
        <li><a href="#experimental-design">Experimental Design</a></li>
        <li><a href="#objectives">Objectives</a></li>
        <li><a href="#sucess-criteria">Success Criteria</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#ui-deployment">UI Deployment</a></li>
      </ul>
    </li>
    <li>
      <a href="#technologies-used">Technologies Used</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- PROJECT DESCRIPTION -->
## Project Description

<p align="center">
  <img 
    width="514"
    height="317"
    src="https://fontspool.com/img/fonts/img-1602929103184.png"
  >
</p>


For this project we conducted sentiment analysis which helped us decipher the customers' reviews and their purchasing behaviors. This helped in knowing what products are in high demand and what specifications are preferred or not preferred in order to advise their respective brands. It also provided insights on how we can better the customer service within the business. We began by performing data cleaning after which we performed univariate and bivariate exploratory data analysis. This enabled us gain insights into the relationships between the various variable we have in the data. We then implemented 3 models i.e. Textblob & Vader which performed the worst with an f1 score of 0.00 and an accuracy score that was pretty low too. Upon implementing XG-boost we gained an improvement in the prediction which could not improve even further upon performing hyperparameter tuning as the accuracy score remained at 59% and so the f1 score at 0.75. Our last model i.e. Bidirectional Encoder Representations from Transformers performed best after merging the sentiments with an accuracy score of 90% and an f1 score of 0.96 


### Aspect Based Sentiment Analysis (ABSA)

ABSA was used to extract customer opinion on a bought product.



### BERT Sentiment Analysis

The BERT model was used to perform sentiment analysis on the product reviews.



<p align="right">(<a href="#top">back to top</a>)</p>

<!-- EXPERIMENTAL DESIGN -->
### Experimental Design

1. Data sourcing/loading 
2. Data Understanding 
3. Data Relevance
4. External Dataset Validation
5. Data Preparation
6. Univariate Analysis
7. Bivariate Analysis
8. Multivariate Analysis
9. Modelling (TextBlob, VADER, XGBoost and BERT)
10. Implementing the solution
11. Challenging the solution
12. Conclusion
13. Follow up questions


<!-- Objectives-->
### Objectives

#### Main Objective: 
To recommend to Amazon the best products to stock in order to meet the customer's taste and needs.

#### Specific Objectives:
To identify and analyze customer opinion on available products.

To develop a system that seeks to forward customers feedback to the brand and predict customer sentiments.

To categorize the products’ reviews


<!-- SUCCESS CRITERIA-->
### Success Criteria

Our study will be considered successful if we are able to meet the objectives.


<!-- DATASET -->
### Dataset

1. [Amazon Fashion Product Reviews](https://nijianmo.github.io/amazon/index.html)
2. [Amazon Fashion Product Metadata](https://nijianmo.github.io/amazon/index.html)
3. [Amazon Fashion Product Ratings](https://nijianmo.github.io/amazon/index.html)



<!-- UI Deployment -->

### UI Deployment

The final project user interface was deployed in [link].

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- TECHNOLOGIES USED -->

## Technologies Used

The languages/IDEs used in the analysis project were: 

**Languages**

[Python](https://www.python.org)

**Web IDE**

[Google Colaboratory](https://colab.research.google.com/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

Steps to be takes to run the project:

1. Download the data from the specified link.
2. Create a copy of the colab notebook.
3. Run the code.


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

1. The project has been written entirely in python, for beginners, refer to the following docummentation for a better understanding of the code:
<ul>
  
  <li> i) [Python](https://www.python.org)</li>
  <li> ii) [Pandas](https://pandas.pydata.org/)</li>
  <li> iii) [Scikit-learn](https://scikit-learn.org/)</li>
  <li> iv) [Huggingface](https://huggingface.co/docs/transformers/model_doc/bert)</li>
  <li> v) [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/generated/seaborn.barplot.html)</li>
  
</ul>

2. Details on the project context and analysis can be found in the colab notebook.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- LICENSE -->
## License

Distributed under the GPL-3.0 license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
List of helpful resources.

* [ABSA](https://medium.com/analytics-vidhya/aspect-based-sentiment-analysis-a-practical-approach-8f51029bbc4a)
* [BERT](https://huggingface.co/docs/transformers/model_doc/bert)
 
<p align="right">(<a href="#top">back to top</a>)</p>




