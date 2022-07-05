# Amazon Product Reviews Sentiment Analysis


## Introduction

The project described below was done with an aim of understanding customer sentiments about products, and their opinion of purchased products.


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
      <ul>
        <li><a href="#experimental-design">Experimental Design</a></li>
        <li><a href="#sucess-criteria">Success Criteria</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#rpubs-publication">RPubs Publication</a></li>
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
    width="614"
    height="417"
    src="https://fontspool.com/img/fonts/img-1602929103184.png"
  >
</p>


Carrefour Kenya was undertaking a project that would inform the marketing department on the presence of fraudulent sales. They provided a data set from
which the most important data set variables were to be identified. They aslo provided datasets to be used in the formation and analysis of association rules, 
and for anomaly(fraud) detection.

### PCA and Feature Selection

Correct identification of the most important variables from the data set was important as this could reduce the amount of factors to be considered during 
decision making and planning, to only the most relevant factors.

### Association Rules

The insights derived from analysis of association rules were important for planning stocking, advertisements, and promotions (to boost sales). Therefore, highly accurate information was required to prevent incurring losses as a result of remaing with a larger than expected closing stock, especially perishable goods.

### Anomaly Detection

Correct detection of anomalies was of importance as under reporting of sales is a crime, as less taxes will be filed. Plus, this activity, as well as 
over reporting of sales could have lead to incorrect restocking of goods, leading to product shortages, or over stocking respectively. This could have lead to heavy losses being incurred, or a great sales opportunity being missed (limited supply of goods in demand).


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
9. PCA, Feature Selection, Association Rules Analysis and Anomaly Detection.  
10. Implementing the solution
11. Challenging the solution
12. Conclusion
13. Follow up questions

<!-- SUCCESS CRITERIA-->
### Success Criteria

1. Correct identification of the most important variables from the provided sales data.
2. Accurately deriving insights on the most popular items, items that had the highest relationship, and types of customers that were likely to return.
3. Correct identification of anomalies within the provided data.


<!-- DATASET -->
### Dataset

The datasets avialed by the client can be downloaded from these links:
1. [Part 1 and 2](http://bit.ly/CarreFourDataset)
2. [Part 3](http://bit.ly/SupermarketDatasetII)
3. [Part 4](http://bit.ly/CarreFourSalesDataset)



<!-- RPUBS PUBLICATION -->
### RPubs Publication

The final project was published in:
1. [Part 1 and 2](http://rpubs.com/deborah_masibo/mba_pca_feature_selection)
2. [Part 3](http://rpubs.com/deborah_masibo/mba_association_rules)
3. [Part 4](http://rpubs.com/deborah_masibo/mba_anomaly_detection)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- TECHNOLOGIES USED -->

## Technologies Used

The languages/IDEs used in the analysis project were: 

**Languages**

[R](https://www.rdocumentation.org/)

**IDE**

[RStudio](https://www.rstudio.com/)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

Steps to be takes to run the project locally.

### Installation

1. Download and install R base.
* [Windows](https://cran.r-project.org/bin/windows/base/)
* [Mac](https://cran.r-project.org/bin/macosx/)
* [Linux (Debian, Fedora/Redhat, Ubuntu)](https://cran.r-project.org/)

2. Download and install [RStudio](https://www.rstudio.com/products/rstudio/download/).
3. Refer to the [RPubs publication](http://rpubs.com/deborah_masibo/908054).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

1. The project has been written entirely in R, for beginners, refer to the [docummentation](https://www.rdocumentation.org/) for a better understanding of the code.
2. Details on the project context and analysis can be found in the RPubs pulication.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
List of helpful resources.

* [R Docummentation](https://www.rdocumentation.org/)
* [Tutorialspoint](https://www.tutorialspoint.com/r/index.htm)
* [Statistics Globe](https://statisticsglobe.com/r-programming-language)
* [Statistical tools for high-throughput data analysis](http://www.sthda.com/english/wiki/ggplot2-essentials)
* [ggplot2](https://ggplot2.tidyverse.org/)
<p align="right">(<a href="#top">back to top</a>)</p>




