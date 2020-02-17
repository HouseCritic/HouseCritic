# HouseCritic

**HouseCritic** is a meta-learning based and semi-supervised deep neural network in order to estimate a specific user’s satisfying degree for a given housing estate.

Concretely, it first captures the user’s preference and the house’s representation from a collection of extracted features. Then, the user preference is used as the meta-knowledge to derive the parameter weights of the house representation such that we can explicitly model the *selection* causality (the decision-making process for users to choose a house according to their preferences) and accordingly provide a satisfying degree of the given house.

## Structure

Figure 1 (a) shows the architecture of the network. which consists of three components:

1. A *train feature extractor* to capture interactions between the current train and other trains on the same line based on the train state. (see Figure 1(b) for the structure of the component)
2. A *passenger feature extractor* for embedding the upcoming passengers’ information among the passenger state by considering and weighing the ST correlations among all these subsequent stations of the train (the structure of the component can be found in Figure 2).
3. A *fusion network* to fuse the two parts of knowledge and accordingly provide Q-values for actions.

![](https://github.com/HouseCritic/HouseCritic/blob/master/img/1.png)

*Figure 1: Structure of HouseCritic*

## <!--Reference-->

<!--*Zhaoyuan Wang, Zheyi Pan. 2020. Shortening passengers’ travel time: A novel dynamic metro train.*-->

## <!--Author-->

<!--*Zhaoyuan Wang-->*



