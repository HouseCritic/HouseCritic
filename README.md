# HouseCritic

**HouseCritic** is a meta-learning based and semi-supervised deep neural network in order to estimate a specific user’s satisfying degree for a given housing estate.

Concretely, it first captures the user’s preference and the house’s representation from a collection of extracted features. Then, the user preference is used as the meta-knowledge to derive the parameter weights of the house representation such that we can explicitly model the *selection* causality (the decision-making process for users to choose a house according to their preferences) and accordingly provide a satisfying degree of the given house.

## Structure

Figure 1 shows the structure of the **HouseCritic**, which consists of three components:

1. *A* *user module,* which captures the user preferences and generates weights of the house embedding based on the user preferences.
2. *A* *house module,* which embeds features of the housing estate.
3. *A* *selection module*, which obtains the houses' estimated satisfying degree of a user. The component is a Meta-FCN, which uses the house embedding as input and the user preference as meta-knowledge (weights). As a result, the satisfying degree can be estimated by modeling the *selection* causality between the user and the housing estate.

![](https://github.com/HouseCritic/HouseCritic/blob/master/img/1.png)

*Figure 1: Structure of HouseCritic*

## <!--Reference-->

<!--*Zhaoyuan Wang, Zheyi Pan. 2020. Shortening passengers’ travel time: A novel dynamic metro train.*-->

## <!--Author-->

<!--*Zhaoyuan Wang-->



