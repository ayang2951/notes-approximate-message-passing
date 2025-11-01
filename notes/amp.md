_Now that we have an understanding of the breakdown of "traditional" methods in high-dimensional settings, we now look at one particular class of algorithms that are useful in a variety of problems._

## Section 2: AMP Algorithms

Having motivated high-dimensional methods in the previous section, we now discuss a particular class of algorithms called <i>approximate message passing</i>. We first introduce its usefulness and a few key characteristics, then discuss the specifics of how the algorithm is structured.

### Benefits and Uses of AMP

At a glance, approximate message passing is a class of iterative algorithms that can be used for studying various high dimensional estimation problems, including high-dimensional linear regression & compressed sensing&mdash;which is where AMP was introduced in statistics by Donoho in 2009&mdash;and phase retrieval, and low-rank estimation problems which encompasses sparse PCA and applications in wireless communications.

There are two features of AMP algorithms that make them attractive:

<ol type="i">
  <li>First, AMP is easily tailored to take advantage of prior information on the estimand. 
  
  For example, if we expect the signal to be sparse, AMP would be a good choice of method. Consider a rank-1 matrix estimation problem where we observe a matrix
  $$Y = \frac{\lambda}{n} v v^T + W,$$
  where we want to estimate $v$ and $W$ is some noise matrix. A typical method may be to use power iteration and use the leading eigenvector. However, if we know $v$ to be sparse, power iteration isn't optimal, as this sparsity constraint would be difficult to enforce. However, AMP makes this easy.</li>
  <li>Second, AMP gives precise asymptotic guarantees in the "large system limit", i.e. when the dimension ($p$) of the estimand grows with the observations ($n$): 
  $$n, p \rightarrow \infty \quad \text{and} \quad \frac{n}{p} \rightarrow \delta \in (0, \infty).$$ 
  In the previous case, the dimension is the same as the size of problem, so clearly $\delta = 1$.

  A natural choice in high dimensional linear regression is the LASSO estimator, which uses iterative soft thresholding to produce an estimate. Here, the AMP iteration does not aim to "compete" with iterative soft thresholding as an estimator; its function is more as analysis of the LASSO estimates. It turns out that AMP may be set up to converge to a fixed point that is guaranteed to be a LASSO solution&mdash;hence, the exact asymptotics that AMP gives us can also be "pushed" onto the LASSO estimate.</li>
</ol>

The two features listed above make AMP useful in a variety of settings for a variety of goals:

<ol type="i">
  <li>Estimation: AMP can be used just for obtaining estimates. It is generally a polynomial-time algorithm and conjectured to be optimal for poly-time algorithms for many problems, making it a good choice amongst the computationally feasible estimators.</li>
  <li>Asymptotics: AMP, as described above, can be used for getting precise asymptotics for many different kinds of estimators.</li>
  <li>Stat-Comp Gaps: AMP can be used to help understand <i>statistical-computational gaps</i> (highly unreasonably called a computational-statistical gap in more computation-focused fields). In many high-dimensional settings, regimes where estimation or testing are <i>information-theoretically possible</i> and regimes where estimation or testing are <i>computationally feasible</i> do not match. For example, consider the linear regression problem
  
  $$y = \beta_0 x + \epsilon.$$
  We may ask how much data is necessary and/or sufficient to recover $\beta_0$: call this $n^\*\_{stat}$. We may then ask how much data is necessary and/or sufficient to recover $\beta_0$ using a polynomial-time algorithm&mdash;call this $n^\*\_{comp}$ (polynomial time algorithms are considered to be the "boundary" of sorts for computational feasibility. Problems that cannot be solved with a poly-time algorithm are called computationally hard). These need not match, and when they don't, there exists a stat-comp gap.</li>
  <li>Lower Bounds: We can use AMP to obtain lower bounds on the estimation error for first order methods. As stated above, it is conjectured that AMP is optimal amongst poly-time algorithms for many different problems, so understanding how well AMP can do gives us some understanding of what poly-time algorithms can achieve in general.</li>
</ol>

### History, Background, and Derivation of AMP

We start with a brief history of AMP and its origins.


### Symmetric AMP Abstract Recursion

We will start with what we call the "abstract AMP recursion" specifically for a symmetric matrix as input. This algorithm is not to be used as a solver for any particular problem with a symmetric matrix, but it provides us with:
<ol type="i">
  <li>A first look at the components of an AMP algorithm, including a Lipschitz function used as a "denoiser" and a memory term called the "Onsager correction". This is what gives us the exact asymptotics for the iterates.</li>
  <li>A starting point for constructing and analyzing AMP algorithms. The form given in the abstract recursion can be used to prove asymptotics for true algorithms: the latter may be molded into the form of the abstract recursion and hence analyzed.</li>
  <li>A precursor to other abstract AMP recursions that can be used to handle variants of low-dimensional estimation problems, such as where the input is an asymmetric matrix, or in generalized linear models.</li>
</ol>

We again emphasize that this is not to be used on a specific problem, but rather can be viewed simply as a result on random matrices. The main result will be that the empirical distributions of the components of the iterates will be asymptotically Gaussian, and that the variance of this Gaussian distribution will be given by the <i>state evolution</i>.

After we introduce and analyze this abstract recursion, we will look at the connects of the abstract algorithm to specific AMP algorithms for glms and low-rank matrix estimation problems.

We start with a few definitions for notation and background.

<div class="callout definition"><span class="label">Definition: Gaussian Orthogonal Ensemble</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
A symmetric matrix $W(n) \in \mathbb R^{n \times n}$, denoted as such to emphasize its dependence on $n$, is called a <em><i>Gaussian orthogonal ensemble</em></i> if 
$$W_{ij} \overset{iid}\sim \mathcal N(0, \frac{1}{n}) \text{ for all } 1 \leq i < j \leq n \qquad \text{and} \qquad W_{ii} \overset{iid}\sim \mathcal N(0, \frac{2}{n}) \text{ for all } i \in \{1, 2 \ldots n\}.$$

Hence, we have that $W \overset{d}= G + G^\top$, where $G \in \mathbb R^{n \times n}$ is such that that 
$$G_{ij} \overset{iid}\sim \mathcal N(0, \frac{1}{2n}) \text{ for all } 0 \leq i, j \leq n.$$
</div>

We now introduce an important property of GOE matrices.

<div class="callout proposition"><span class="label">Proposition: GOE and Orthogonal Matrices</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $W(n)$ be a GOE matrix, and let $Q \in \mathbb R^{n \times n}$ be orthogonal. Then 
$$QWQ^\top \sim \text{GOE}(n).$$ 
</div>

We define the $(n, 2)$ inner product and norm, which will be notationally convenient for us for the main AMP results.

<div class="callout definition"><span class="label">Definition: $n$-Inner Product & $(n, r)$-Norm</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $u, v \in \mathbb R^n$. We define the $n$-inner product to be the scaled Euclidean norm:
$$\langle u, v\rangle_n \triangleq \frac{1}{n}\langle u, v\rangle = \frac{1}{n} \sum_{i = 1}^n u_i \cdot v_i.$$
We define the $(n, r)$ norm to be
$$\Vert v \Vert_{n, r} = \frac{1}{n} \Vert v \Vert_r = \left (\frac{1}{n} \sum_{i = 1}^n |v_i|^r\right)^\frac{1}{r}.$$

</div>

We are now ready to introduce the abstract AMP recursion. We will first formalize the recursion itself, then informally state the key result for the asymptotics of this recursion, then formalize these statements in the next section with more care and technicality.

<div class="callout remark"><span class="label">Remark: Asymptotics of Iterations</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
It is important to note that the asymptotics we will analyze will be for <i>each</i> iterate $k \in \mathbb N$ as $n \rightarrow \infty$, not convergence as the number of iterates increases.
</div>

#### Symmetric AMP Recursion

We have a symmetric random matrix $W$.

Let $\\{f\_k\\}\_{k = 0}^\infty$ be a sequence of Lipschitz functions that take vector input and act componentwise. Let $m^{-1}(n) \triangleq 0$, and let $h^0$ be an initializer. For each $k \in \mathbb N$, define
$$m^k = f\_k(h^k), \qquad b\_k = \frac{1}{n} \sum_{i = 1}^n f\_k^{'}(h\_i^k), \qquad h^{k + 1} = Wm^k - b_k m^{k - 1}.$$
As stated above, we are interested in the convergence of each of these iterates as $n \rightarrow \infty$. We will now state the key result, informally, of this convergence.


<div class="callout theorem"><span class="label">Theorem: State Evolution of AMP Iterates (INFORMAL)</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $W$ be a GOE matrix, and let $h^0$ be an initializer independent of $W$. Let $m^{-1}(n) \triangleq 0$ and $\tau_1^2 \triangleq \lim_{n \rightarrow \infty} \frac{1}{n} \Vert f_0(h^0) \Vert_2^2$, which we assume to be finite.

Then for each $k \in \mathbb N$, for the AMP iterates defined by
$$m^k = f\_k(h^k), \qquad b\_k = \frac{1}{n} \sum_{i = 1}^n f\_k^{'}(h\_i^k), \qquad h^{k + 1} = Wm^k - b_k m^{k - 1},$$
the empirical distribution of $h^k$ (we shall denote this $\hat P_{h^k}$) converges to $\mathcal N(0, \tau_k^2)$ in Wasserstein distance (this will be stated more precisely in the formal theorem statement), where the variance is defined by the recursion
$$\tau_{k + 1}^2 = \mathbb E[(f_k(Z_k)^2)],$$
where $Z_k \sim \mathbb N(0, \tau_k^2)$. This recursion for the variance terms is called the <em><i>state evolution</em></i>.
</div>

<details class="collapsible">
  <summary>Heuristic Proof of Theorem 2.1.</summary>
  <div class="collapsible__content">
We start with the first iterate: show $\hat P_{h^1} \rightarrow \mathcal N(0, \tau_1^2)$ where we have by definition that 
$$h^1 = Wf_0(h^0) \quad \text{and} \quad \tau_1^2 \triangleq \lim_{n \rightarrow \infty} \frac{1}{n} \Vert f_0(h^0) \Vert_2^2 < \infty.$$

Conditioned on $h^0$, because $h^0 \perp W$, we have that (abusing notation for distribution)
$$W f_0(h^0) \bigg\lvert_{h^0} \overset{d}= \mathcal N \left(0, \ \frac{1}{n}\sum_{i = 1}^n f_0(h^0_i) \mathbb I_n + \frac{1}{n} f_0(h_0) f_0(h_0)^\top\right) \qquad \text{ for each } n \in \mathbb N.$$
We can first prove this above result on the finite-sample distribution.
<details class="collapsible">
<summary>Computing the Finite-Sample Distribution.</summary>
<div class="collapsible__content">
Recall that for $W \sim \text{GOE}(n)$, $W \overset{d}= G + G^\top$ where $G_{ij} \overset{iid}\sim \mathcal N(0, \frac{1}{2n})$. Hence, we have that
$$W f_0(h^0) \bigg\lvert_{h^0} \overset{d}= (G+G^\top)f_0(h^0)\bigg\lvert_{h^0} = [G f_0(h^0) + G^\top f_0(h^0)] \bigg\lvert_{h^0}.$$
We now simply compute the mean and covariance matrix of the components of the resultant vector. 
$$\mathbb E_{W | h^0}[(W f_0(h^0))_i] = \mathbb E_{W|h^0}[(f_0(h^0)^\top G_{i, \boldsymbol{\cdot}}) + (f_0(h^0)^\top G_{\boldsymbol{\cdot}, i})] = 0,$$
since this is the sum of $2n - 1$ independent Gaussians with mean 0, variance $\frac{1}{2n}$ and 1 Gaussian with mean 0, variance $\frac{2}{n}$ because element $G_{ii}$ was repeated.

Next, we compute the variance. Let $0\leq i, j \leq n$, and we compute $\text{Cov}((W f_0(h^0))_i, (W f_0(h^0))_j)$.
$$equation$$
Hence, we obtain the form above for $n$ finite.
</div>
</details>
We now take $n \rightarrow \infty$ to see the asymptotic result. We have that 
$$\frac{1}{n} \Vert f_0(h^0) \Vert_2^2 \rightarrow \tau_1^2$$
by assumption, and for "reasonable" $f_0(h^0)$, we have that the element
$$(\frac{1}{n} f_0(h^0) f_0(h^0)^\top)_{ij} = \frac{1}{n} f_0(h^0)_i \cdot f_0(h^0)_j \rightarrow 0 \quad \text{as } n \rightarrow \infty.$$
Hence, we have that for the first iterate,
$$\hat P_{h^1} = \hat P_{W h^0} \rightarrow \mathcal N(0, \tau^2_1).$$

The next step is to prove the same for $h^2$, but the challenge now is that $W$ is not independent of $h^1$:
$$h^2 = W m^1 - b_1 m^0 \qquad \text{where } m^1 = f_1(h^1).$$
However, consider a new matrix $\tilde W \sim \text{GOE}(n) \perp m^1$. Then we have that $\tilde Wm^1$ is Gaussian, and that, for $\hat P_n$ denoting the empirical distribution, 
$$\hat P_n(\tilde W m^1) \bigg \lvert_{m^1} \rightarrow \mathcal N(0, \tilde \tau_2^2),$$
where 
$$\tilde \tau_2^2 \overset{(1)}= \lim_{n\rightarrow\infty} \frac{1}{n} \Vert m^1 \Vert_2^2 \overset{(2)}= \lim_{n \rightarrow\infty} \Vert f_1(h^1) \Vert_{n, 2} \overset{(3)}= \mathbb E[(f_1(Z_1))^2] \overset{(4)}= \tau_2^2,$$
where, as before, $Z_1 \sim \mathcal N(0, \tau_1^2)$.

Walking through the equalities, we have that $(1)/(2)$ follow from the same argument as was used in the first iteration, the proof of $(3)$ requires many, many technical steps and can be ignored for now (note that we <i>cannot</i> simply apply our typical LLN argument), and $(4)$ is from our definition of $\tau_k^2$ in the state evolution recursion&mdash;how we defined $\tau_2^2$.

At a high-level, the role of the Onsager correction term $b_1 \cdot m^0$ is the "cancel out" the dependence exactly in the limit. It ensures that at, for example, the second time step, $h^2$ asymptotically has the same distribution as $\tilde W m^1$.

Having proved the first two time steps (one with the dependence that needed correction), we proceed to the inductive step: showing that $\hat P_n(h^{k + 1}) \rightarrow \mathcal N(0, \tau_{k}^2)$.

We have $h^{k + 1} = W f_k(h^k) - b_k \cdot f_{k - 1}(h^{k - 1})$. Let us define $\mathscr S_k \triangleq \sigma(\{h^1, h^2 \ldots h^k; h^0\})$ as the $\sigma$-field generated by all previous iterates of $h$, $b$, $m$, but notably does not include $W$.
</details>

<div class="callout proposition"><span class="label">Proposition: Prop Name</span><br/>
<hr style="height:0.01px; visibility:hidden;" />

</div>