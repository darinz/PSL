# Decision Trees

## Decision Trees: nonlinear classifier

<img src="./img/nonlinear-classifier.png" width="650px">

## Decision Trees: canonical situation

- No linear separation line
- Want to divide input space into "regions"
- Can do this by dividing input space into disjoint regions $R_i$

$$\mathcal{X} = \bigcup_{i=0}^{n} R_i$$

s.t.

$$R_i \cap R_j = \emptyset \text{ for } i \neq j$$

## Recursively splitting regions

- Parent region $R_p$
- "Children" regions $R_1$ and $R_2$
- Split on feature $X_j$

$$R_1 = \{X \mid X_j < t, X \in R_p\}$$

$$R_2 = \{X \mid X_j \geq t, X \in R_p\}$$

Split 1:

<img src="./img/split_1.png" width="550px">

Split 2:

<img src="./img/split_2.png" width="550px">

Split 3:

<img src="./img/split_3.png" width="550px">

