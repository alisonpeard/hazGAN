## Return periods calculation
Following [this page](https://georgebv.github.io/pyextremes/user-guide/6-return-periods/), return periods are calculated as following: let $x$ be the maximum wind speed of a storm and we rank the $N$ storms from largest to smallest where the rank of the biggest storm is $r(x)=1$. 

Then the exceedance probability $$S(x) = \mathbb{P}(X\geq x)=\frac{r(x)}{N+1}$$.

If we have $\lambda=17$ storms in a year, the return period is given by
$$\text{RP} = \frac{1}{\lambda S(x)}.$$

To have return period greater than 1-year,
$$ 
\begin{align*}
\text{RP} = \frac{1}{\lambda S(x)} &\geq  1 \\
\lambda S(x) &\leq 1 \\
S(x) &\leq \frac{1}{\lambda} \\
\frac{r(x)}{N+1} &\leq \frac{1}{\lambda} \\
r(x) &\leq \frac{N+1}{\lambda} \\
\end{align*}
$$

so for a dataset of $1200$ events with 17 events per year, we need $x$ to be in the $1201 / 17 = 70.64$ largest events in the dataset, so expect $1130$ events to have RP less than 1, with some variation for averaging ranks across ties.

Actually, the dataset has $1228$ storms, and $1159$ are assigned a RP less than 1, while $69$ greater than 1. So seems approximately correct.