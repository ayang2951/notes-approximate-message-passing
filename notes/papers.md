## Equivalence of LDP and AMP in Low-Rank Matrix Estimation

PAPER: "Equivalence of approximate message passing and low-degree polynomials in rank-one matrix estimation," Montanari, A., Wein, A.

### Motivation

We have hopefully developed a thorough understanding of the importance of computationally feasible algorithms in high dimensional statistics. The focus of this section will be to motivate the new lines of work arising within this body of literature.

The stat-comp gap community comprises people studying various classes of poly-time (or wider computationally feasible) algorithms, which includes but is not limited to approximate message passing, low-degree polynomials, statistical query models, and sum-of-squares. For such classes of algorithms, what we have observed is that papers are published on the fundamental barriers that appear in statistical tasks such as estimation or testing, and it is subsequently discovered that these barriers match across the different classes. This particular paper demonstrates this barrier-matching for low-degree polynomials and approximate message passing in the rank-one matrix estimation problem.

The matching of these lower bounds lends credence to the belief that these barriers are fundamental for computationally feasible algorithms, and an interest has been developed to pivot to <i>actively</i> demonstrating that the <i>classes</i> of algorithms can be "reduced" in some way to one another, in the average-case, for a particular problem. This paper contributes to this approach, which the authors of this paper say "[trims] the space of possible algorithmic choices"

### Problem Setup

We will focus on the rank-one estimation problem, where we have a noisy matrix observation $Y \in \mathbb R^n$:
$$Y = \frac{1}{\sqrt n} \theta \theta^\top + Z,$$
where $\theta_i \overset{iid}\sim \pi_{\Theta}$ and $Z \sim \text{GOE}(n)$, $\pi_{\Theta}$ being a prior that does not depend on $n$.

This is a problem we have seen quite a bit of before, there being a section dedicated to it in the AMP tutorial. Note that the scaling is slightly different: $\frac{1}{\sqrt n}$ instead of $\frac{1}{n}$.

Our aim will naturally be to estimate $\theta$.


### Bayes & AMP Background

A few quantities introduced here originate from statistical physics, for which I have essentially zero understanding and intuition for. We will introduce a sort of potential function which represents some form of "free-energy landscape."

Define for $\pi_{\Theta} \in \mathscr P(\mathbb R)$ the set of probability distributions on $\mathbb R$ and $q \in \mathbb R^+, b \in \mathbb R$, 
$$\Psi(q; b \pi_{\Theta}) \triangleq \frac{1}{4} \big(q - \mathbb E[\Theta^2]\big)^2 - \frac{1}{2} bq + \text{I}(Y_q; \pi_{\Theta}),$$
where $\text{I}(Y_q; \pi_{\Theta})$ is the mutual informtion between $\Theta$ and $Y_q \triangleq \sqrt{q} \Theta + G$, and $(\Theta, G) \sim \pi_{\Theta} \otimes \mathcal N(0, 1)$. The mutual information in a form that signals the "temporal" aspect of the data generation:
$$\text{I}(Y_q, \pi_{\Theta}) \triangleq - \mathbb E\bigg[\log \frac{dp_{Y_q | \Theta}}{dp_{Y_q}}\bigg],$$
i.e. $Y_q \big\vert_{\Theta = \theta} \sim \mathcal N(\sqrt q \cdot \theta, 1)$. 

We need not worry too much about the quantity $b$, it represents some sort of mean shift for the prior expectation, and we will essentially only work with the quantity above for when $b = 0$. $q$ is the true quantity of interest, and the following definitions will aid us in defining the information-theoretic and computation lower bounds for AMP & LDP.

Let
$$q_{\text{bayes}}(\pi_{\Theta}) \triangleq \arg\min_{q \geq 0} \Psi(q; 0, \pi_{\Theta})$$
and
$$q_{\text{amp}}(\pi_{\Theta}) \triangleq \inf \\{q \geq 0 : \Psi'(q, 0, \pi_{\Theta}) = 0, \Psi''(q, 0, \pi_{\Theta}) \geq 0\\}.$$

The following theorem gives us the lower bounds on MSE.

<div class="callout theorem"><span class="label">Theorem: (1.1) Lower Bounds on MSE</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
In the rank-one matrix estimation model introduced above, assume that $\mathbb E[\Theta] \neq 0$.

Then for the universe of estimators $\hat \theta$, we have that
$$\lim_{n \rightarrow \infty} \inf_{\hat\theta} \frac{1}{n} \mathbb E[ \Vert \hat\theta(Y) - \theta \Vert_2^2] = E[\Theta^2] - q_{bayes}(\pi_{\Theta}).$$

And for the optimal AMP algorithm (which we know is the Bayes denoiser, but we will review this in the section below), we have that
$$\lim_{n \rightarrow \infty} \frac{1}{n} \mathbb E[ \Vert \hat\theta^{amp}(Y) - \theta \Vert_2^2] = E[\Theta^2] - q_{amp}(\pi_{\Theta}).$$
</div>

As we can see, in interest of minimizing the MSE, $q_{bayes}$ or $q_{amp}$ should be large. For the quantity $q_{bayes}$, we see that $q$ achieves the global minimum of the function $\Psi$, which as we recall represents a free-energy landscape. Whether AMP can achieve this same minimum depends on whether it is "impeded by a free-energy barrier," as stated in the survey paper by Wein. We recll Cindy's remark that when the problem is not completely nice, the minimum may not be unique, and whenever this is the case, AMP will converge to the worst fixed point. $q_{amp}$ being a quantity we want to maximize, taking the infimum represents this unfortunate reality exactly.

We may now review the AMP results for this problem. Consider our classical AMP iteration of the form
$$x^{t + 1} \triangleq \frac{1}{\sqrt n} Y \cdot f_t (x^t) - f_{t - 1}(x^{t - 1}) \cdot \frac{1}{n} f'_t(x^t).$$
Recall that our classical results on the state evolution hold for $\{f_t\}$ are Lipschitz functions. Is it surprising if we claim that the state evolution results apply similar for polynomial functions? For a reference on this, check out Bayati, Lelarge, and Montanari 2015, "Universality in polytope phase transitions and message passing algorithms."

We're familiar with the state evolution recursions
$$\mu_{t + 1} \triangleq \mathbb E[\Theta f_t(\mu_t \Theta + \sigma_t G)]$$
and
$$\sigma^2_{t + 1} \triangleq \mathbb E[f_t^2 (\mu_t \Theta + \sigma_t G)]$$
where $(\Theta, G) \sim \pi_{\Theta} \otimes \mathcal N(0, 1)$.

When we use what we know is the sequence of optimal denoisers, i.e. the posterior expectation with constant $q_{t + 1} = \mathbb E\bigg[ \bigg(\mathbb E\big[\Theta | q_t \Theta + \sqrt{q_t} G\big]\bigg)^2\bigg]$, we have $f_t(x) = E[\Theta | q_t \Theta + \sqrt{q_t} G]$ and
$$q_{t+1} = \mu_{t+1} = \sigma^2_{t+1}.$$

Defining $\{q_t\}$ this way, we have the following lower bound result on the entire class of AMP algorithms.

<div class="callout theorem"><span class="label">Theorem: (2.2) Lower Bounds on the Class of AMP Algorithms</span><br/>
<hr style="height:0.01px; visibility:hidden;" />

Suppose the second moment of $\pi_{\Theta}$ is finite. Then, for any AMP algorithm, for any fixed iteration $t$, we have that
$$\lim_{n \rightarrow \infty} \frac{1}{n} \mathbb E[\Vert \hat\theta^{t}(Y) - \theta] \Vert_2^2 \geq \mathbb E[\Theta^2] - q_{t + 1}.$$

Further, there exists a sequence of AMP algorithms that approach this lower bound arbitrarily closely.

The fixed points of this recursion match the <i>stable points</i> of the potential function $\Psi$, i.e. the fixed points such that
$$q^* = \mathbb E\bigg[ \bigg(\mathbb E\big[\Theta | q^* \Theta + \sqrt{q^*} G\big]\bigg)^2\bigg]$$
are the points such that
$\partial_q(\Psi(q, 0, \pi_{\Theta})) = 0$.
</div>

This concludes the necessary background on the Bayes and AMP results, and we move on to discussing the LDPs.


### Low-Degree Polynomial Background

For $\hat \theta : \mathbb R^{n \times n} \rightarrow \mathbb R^n$, we call $\hat\theta$ a low-degree polynomial estimator of $D$ degree if the coordinates of the function are polynomials in the entries of $Y$ of at most degree $D$.

Let us define $\mathbb R[Y]\_{\leq D}$ to be the space of all polynomials of degree $D$ in the entries of $Y \in \mathbb R^{n \times n}$. Let us further define the space
$$\text{LD}(D, n) \triangleq \{\hat\theta \in \mathbb R^n : \forall i \in [n], \hat\theta_i \in \mathbb R[Y]_{\leq D}\},$$
i.e. all polynomials whose coordinates are degree at most $D$.

In this paper, we will be working only with constant-degree LDPs, contrasting the perhaps more conventional degree $\textbf{O}(\log n)$ described in the survey paper by Wein.

<div class="callout remark"><span class="label">Remark: Degree of the LDPs</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
That this result is only stated to hold for constant degree polynomials is not believed to be fundamental&mdash;and in fact the result is believed to hold for up to degree $\textbf{O}(n^c)$ for $c \in (0, 1)$&mdash;but rather an assumption made for a slick proof.  
</div>


### Equivalence Results for AMP and LDP

We are now prepared to state the main result of equivalence of the paper. 

<div class="callout theorem"><span class="label">Theorem: Equivlence of AMP and LDP</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Suppose as before that $Y = \frac{1}{\sqrt n} \theta\theta^T + Z$ where $\theta_i \sim \pi_{\Theta}$ which is independent of $n$, has nonzero mean, and is subgaussian. Then 

<ol type="i">
  <li>For any $\epsilon > 0$, there exists a degree $D_\epsilon < \infty$ and an associated sequence of estimators $\hat\theta_\epsilon \in \text{LD}(D_\epsilon, n)$ such that
  $$\lim_{n \rightarrow \infty} \mathbb E\big[\Vert \hat\theta_\epsilon(Y) - \theta \Vert_2^2 \big] \leq \mathbb E[\Theta^2] - q_{amp}(\pi_{\Theta}) + \epsilon.$$
  </li>
  <li>Conversely, for any constant $D < \infty$,
  $$\lim_{n \rightarrow \infty} \inf_{\hat\theta \in \text{LD}(D, n)} \frac{1}{n} \mathbb E \big[\Vert \hat\theta(Y) - \theta \Vert_2^2\big] \geq \mathbb E[\Theta^2] - q_{amp}(\pi_{\Theta}).$$
  </li>
</ol>
</div>

In words, Bayes AMP with a constant number of runs matches performance with constant-degree low-degree polynomials. The theorem states that, for all $\epsilon > 0$, there exists a constant $D_\epsilon$ where a sequence of low-degree estimators well approximates the performance of the optimal (Bayes) AMP with at most an additive error $\epsilon$. Further, for any degree $D = \textbf{O}(1)$, the performance of any degree-$D$ estimator $\hat\theta(Y)$ is lower bounded by that of Bayes AMP, i.e.
$$\frac{1}{n} \mathbb E[\Vert \hat\theta(Y) - \theta \Vert_2^2] \geq \mathbb E[\Theta^2] - q_{amp}(\pi_\Theta) - \textbf{o}_n(1).$$

We may now discuss the proofs of these two bounds.

#### The Upper Bound

In broad strokes, the claim is that we can select, for any $t$, for any $\epsilon_0 > 0$, we may find a sequence of polynomials $\{f_s\}\_{0 \leq s \leq t}$ of degrees $\\{D_{\epsilon_0}^s\\}_{0 \leq s \leq t}$ such that
$$|\mu - q_t| \leq \epsilon_0 \qquad \text{and} \qquad |\sigma^2_t - q_t| \leq \epsilon_0.$$
The resultant low-degree polynomial estimator would be $\hat\theta^t(Y) = f_t(x^t)$ where $x^t$ is as defined above in the AMP recursion and has degree at most $(D_1 + 1)(D_2 + 1) \ldots (D_t + 1)$.

In essence, we may use low-degree polynomial functions to approximate the Bayes denoiser (not a polynomial) very well in the sense of the state evolution parameters (through which the MSE is defined), which is not a shocking result.

#### The Lower Bound

The lower bound is more technically challenging (requiring some graph theory) and, perhaps, innovative. It relies on representing low-degree polynomials as functions on graphs, then claiming that the asymptotically-best estimators are only trees. AMP is then reduced to message passing/belief propogation, which is represented via trees. Through trees, then, we can characterize the optimal performance amongst low-degree polynomials and AMP algorithms.

Let $\mathcal T_{\leq D}$ be the equivalence class of rooted trees up to root-preserving isomorphism$ with at most $D$ edges, i.e. elements in $\mathcal T_{\leq D}$ are distinct tree-shapes with the same root.

At a high-level, for any tree $T \in \mathcal T_{\leq D}$, we may define one associated polynomial $\mathscr F_T(Y)$, and we will consider <i>non-reversing labelings</i> over this tree.

<div class="callout definition"><span class="label">Definition: Non-Reversing Labels</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
A labeling $\phi$ of a tree $T = (V, E) \in \mathcal T\_{\leq D}$ is called <em><i>non-reversing</em></i> if, for all distinct pairs of vertices $u, v \in V$ where $\phi(u) = \phi(v)$, either $\text{d}_T(u, v) > 2$ or $u$ and $v$ are children of some common vertex.

Call $\text{nr}(T)$ the set of all non-reversing labelings of $T \in \mathcal T_{\leq D}$.
</div>

For each $T \in \mathcal T_{\leq D}$, we will define a specific polynomial: let
$$\mathscr F_T(Y) \triangleq \frac{1}{\sqrt {|\text{nr}(T)}|} \sum_{\phi \in \text{nr}(T)} \prod_{(u, v) \in E(T)} Y_{\phi(u), \phi(v)}.$$

We see that the polynomial is a function of the elements of $Y$ where there exists an edge between the vertices defined by the indices, of which there are only $D$.

Now, we define the class of trees that will represent the LDPs.

<div class="callout definition"><span class="label">Definition: Class of Tree-Structured Polynomials</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let coefficients $\hat p_T \in \mathbb R$, define the class 
$$\mathbb R[Y]^{\mathcal T}_{\leq D} \triangleq \left\{p(Y) : p(Y) = \sum_{T \in \mathcal T_{\leq D}}\hat p_T \cdot \mathscr F_T(Y)\right\}$$
as the class of tree-structured polynomials.
</div>

Consider without loss of generality the estimation of the first coordinate of $\theta$, i.e. $\theta_1$. the following proposition posits that the degree-$D$ tree-structured polynomials perform just as well as the class of all degree-$D$ polynomials.

<div class="callout proposition"><span class="label">Proposition: Optimality of Tree-Structured Polynomials</span><br/>
<hr style="height:0.01px; visibility:hidden;" />
Let $\pi_{\Theta}$ as before have all-finite moments. Let $\Psi(\theta_1)$ to be a measurable function that also has all-finite moments. For any fixed $D$, there exists a fixed choice of coefficients $\{\hat p_T\}_{T \in \mathcal T_{\leq D}}$ (independent of $n$) such that, the associated polynomial $p_n \in \mathbb R[Y]^{\mathcal T}_{\leq D}$ defined, like above, as $p(Y) = \sum_{T \in \mathcal T_{\leq D}}\hat p_T \cdot \mathscr F_T(Y)\right$, satisfies
$$\lim_{n \rightarrow \infty} \mathbb E[(p(Y) - \psi(\theta_1))^2] = \lim_{n \rightarrow \infty}\inf_{\tilde p \in \mathbb R[Y]_{\leq D}} \mathbb E[(\tilde p(Y) - \psi(\theta_1))^2].$$

Proof aside, it remains to connect these tree-based polynomials to AMP. This will be done by examing AMP through its origins of message passing on graphs.

</div>

<div class="callout theorem"><span class="label">Theorem: Theorem Name</span><br/>
<hr style="height:0.01px; visibility:hidden;" />


</div>

<ol type="i">
  <li>.</li>
  <li>.</li>
</ol>