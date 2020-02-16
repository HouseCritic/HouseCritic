# HouseCritic

**HouseCritic** is a meta-learning based deep neural network which can estimate the housing estate’s satisfying degree of the user. 	

In the beginning, it can capture users’ preference and houses’ representation from the extracted features. Then, user preference is used as the meta-knowledge to derive the parameter weights of house representations. Lastly, we can explicitly model the selection causality from house representation and accordingly provide a satisfying degree of a given house.

## Structure

**HouseCritic** consists of three components: 

1. a *user module* generates weights of the house embedding from the user feature which is the meta-knowledge in Component 3.
2. a *house module* embeds features of the house.
3. a *selection module*  obtains the houses' estimated satisfying degree of a user. The component is a Meta-FCN which uses the house embed as input and meta-knowledge generated from Component 1 as weight. As a result, the satisfying degree can model the “selection” causality between user and house.

Figure 1 (a), (b) and (c) show  the structures of the *user module*, *house module* and *selection module*, respectively.

![](https://github.com/HouseCritic/HouseCritic/blob/master/img/1.png)

*Figure 1: Structure of HouseCritic*

## <!--Reference-->

<!--*Zhaoyuan Wang, Zheyi Pan. 2020. Shortening passengers’ travel time: A novel dynamic metro train.*-->

## <!--Author-->

<!--*Zhaoyuan Wang-->*



