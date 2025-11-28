# Simulation-Based-Inference

## Setup

Simulation Based Inference (SBI) is what it says on the box, attempting to infer something about the world primarily through simulations. 
It is also sometimes called Likelihood Free Inference (LFI) though this is somewhat misleading, since we do have access to the likelihood just not the density, a sampler is an expression of the likelihood though in a somewhat inconvenient format.

Ok, lets be more specific, our goal is to infer a posterior probability distribution $P(\theta | X)$ over parameters given some data where $\theta$ is the parameters and $X$ is the data.
Typically, what we have access to is actually a prior $P(\theta)$ over the parameters (if you think you don't have this, I assure you that you do) and a likelihood $P(X | \theta)$.
Those two parts should describe how our data came about through a two step process:

1. Sample parameters $\theta_i \sim P(\theta)$
1. Sample the data $X_i \sim P(X | \theta_i)$

The goal of SBI is to infer the posterior on $\theta$ given some data $X$ and the two sampling functions above, under the restriction that we can't evaluate the probability densities of those functions (or at least not the likelihood).

There are actually broadly three reasons you would want to use SBI:

1. You can't evaluate your likelihood. This is the core motivation, it comes from very complex simulators that include as much realism as possible. Often such a focus on realism adds so much complexity that a likelihood density becomes impossible to determine.
1. You can evaluate your likelihood, but it is very slow. In this case, evaluating the likelihood involves costly integrals, gigantic matrix inversions, or other compute heavy operations that are impractical to perform the many thousands of times necessary to sample the full distribution.
1. Your likelihood has a large or trans-dimensional number of nuisance parameters. If you only care about a few parameters, you may be able to use SBI to marginalize over all/some of the nuisance parameters preemptively (just note that this likely means you are baking in a prior for those nuisance parameters).

Of course, some mix of these challenges is more likely for any particular problem.

## The four approaches

There are broadly four approaches to SBI used in the community: Neural Likelihood Estimation (NLE), Neural Posterior Estimation (NPE), Neural Joint Estimation (NJE), and Neural Ratio Estimation (NRE).
Though I would prefer to just call them: Likelihood Estimation, Posterior Estimation, Joint Estimation, and Ratio Estimation, since the "Neural" part is just a implementation detail and would be dropped tomorrow if we found something better than neural networks for the task.

In Figure 1 we plot the four different objectives of these SBI approaches.

- The first option NLE (top left) somewhat directly addresses the "Likelihood Free" aspect, since it attempts to approximate the likelihood density. This has some key advantages in that it is independent of the prior, and often (though not always) relatively simple compared to the others.
- The most direct is the NPE (top right) which simply fits a posterior density function so that one can later enter their data $X$ and get a probability density function (PDF) for the parameters of interest.
Note that having $P(\theta | X)$ can sometimes be only half the battle, if $\theta$ has many dimensions then even if we can evaluate it, we still will need to do some work to get marginal distributions.
- The least used is NJE (bottom left) since it requires working with the highest dimensions possible (joined $\theta$ and $X$) which is really hard for density estimation. It also has to be globally normalized.
- The NRE (bottom right) approach has some nice features, it can be easier to train (its just a classifier, see below), letting the prior do a lot of work later on (see the math stuff).

![Comparison of different SBI target distributions](SBIdemo.png)
*Figure 1: Comparison of the different SBI target distributions. Here we see the interplay of the likelihood, prior, and different normalization directions in a very simple inference task.*

The NPE, NJE, and NRE have a major drawback, namely that they bake the prior into their distribution so it is not possible (specifically it is not numerically stable) to change up the prior after the fact, a new SBI model would need to be trained for a new prior.
Another factor is that the likelihood tends to be comparatively simple to model, in our example there is a mean relationship $X=\theta^2$ and after that it's just a Gaussian of constant width.
Since an NLE is trained with the $\theta$ values provided, it is possible to train the NLE from any set of $\theta$ values, so one can use active learning to train more accurately; though there is some element of optimality in choosing the prior since it will give more examples where there ought to be more data.
I should note that while the NRE appears hopeless in that the high values are all at high $X$ which we don't care about, the situation is actually not so bad.
The training would happen in log density space, which handles very large/small values equally, and also the training data would be concentrated in the center where the prior dominates so the NRE would have lots of examples in the right area to constrain it.

Figure 1 shows some cool effects that aren't SBI specific, but are just neat aspects of probability.
Notice how in the Posterior distribution the probability is pulled towards the center, this is because of the prior.
We see how the likelihood looks like it gets thinner for high $\theta$ values, though it is really constant thickness, its just that the $\theta^2$ relationship is getting very steep.
I also think it is neat to see a real difference between $P(\theta, X)$, $P(\theta | X)$, and $P(X | \theta)$ since Gaussians alone are somewhat boring in this regard.

## Example

There are some great examples of SBI being used in astronomy.
[This paper](https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.4440A/abstract) provides a nice overview of techniques, as well as some very simplified examples.

I feel compelled to comment again that the "Neural" part of NJE, NPE, NLE, and NRE is not always (in fact perhaps often not) needed.
Consider the example of photometric redshift estimation as discussed in the [RAIL paper](https://ui.adsabs.harvard.edu/abs/2025arXiv251007370Z/abstract).
It is trivial to determine the redshift of a galaxy from a high resolution, high signal-to-noise spectrum, but often all we get is a few wide integrated bands (photometry).
Trying to estimate these photometric redshifts is hard but not impossible, and we can see in Figure 2 that it broadly works well for a variety of algorithms.

![RAIL paper comparison of photometric redshift estimation algorithms](photoz_specz.png)
*Figure 2: comparison of multiple photometric redshift estimation algorithms. Estimated redshift (y-axis) versus reference redshift (x-axis). Reference: This is Figure 4 in [Zhang et al. 2025](https://ui.adsabs.harvard.edu/abs/2025arXiv251007370Z/abstract)*

While the relations in Figure 2 are quite tight, there is clearly notable scatter for all of them that is significantly away from the 1:1 line.
This is a cause of great consternation in the photo-z community, with many discussions on "de-biasing" the predictions.
Though talk of de-biasing is perhaps focusing too much on the first moment of the data, when there is much more going on.
Even if we could de-bias the mean prediction, clearly there isn't a simple Gaussian scatter that could describe the uncertainty on the photometric redshifts.
But that's ok, we can use the concepts from SBI!
These 2D histograms essentially provide the ~~N~~LE or ~~N~~PE depending on how you normalize it.
You would probably want to get a lot more examples, or apply some smoothing, but trying to use a Neural network is like using a cement mixer to make bread (certainly overkill, and probably not very good in the end anyway).

## Some math

Ok, so we don't **always** need a neural network, but sometimes we do, so how does that work?
The NRE's are a bit different, so we'll start with NPE, NJE, and NLE which are largely the same.
Let's think of our NLE (just switch $X$ and $\theta$ around for NPE and use both for NJE) as a probability density $P(X | \theta, w)$ where $w$ are the network weights.
Our target is the true $P(X | \theta)$ PDF, which we can enforce using the Kullback–Leibler divergence:

$$D_{KL} = \int P(X_i|\theta)\ln\left(\frac{P(X_i|\theta,w)}{P(X_i|\theta)}\right)dX_i$$

Which would be a nasty integral to solve directly, but we can do a good job with a Monte-Carlo estimate of the integral.
To Monte-Carlo an integral, we sample from one component ($P(X_i | \theta)$ in this case) and evaluate the other component:

$$D_{KL} \approx \frac{1}{N}\sum_{X_i\sim P(X_i|\theta)}\ln\left(\frac{P(X_i|\theta,w)}{P(X_i|\theta)}\right)$$

Note that the sum is over samples $X_i\sim P(X_i|\theta)$ which is great!
Since we are allowed to sample $P(X|\theta)$ we just can't evaluate its density.
But the $P(X_i|\theta)$ density still appears as a ratio in the log function, oh no!
Luckily, we don't actually need to know the KL-divergence, we just need to get the network to minimize the KL-divergence.
If we take the gradient $\nabla_{w}D_{KL}$ then the dependence on the true likelihood goes away!
So it is possible to minimize the KL divergence without ever directly evaluating it (wild), and this is actually a pretty powerful observation used elsewhere in machine learning too.

The NRE method works a bit differently, for this you train a classifier.
The classifier learns to distinguish $\{\theta, X\}$ pairs that were simulated together (the two step process at the top) vs random $\{\theta', X'\}$ marginal pairs (just pulled from a large collection).
If you use the binary cross entropy loss then in the limit of lots of examples/training the classifier learns the ratio:

$$r(\theta, X) = \frac{P(\theta, X)}{P(\theta)P(X)}$$

To get the posterior, it turns out that $r(\theta, X)P(\theta)$ is all you need!
This just comes from the chain rule of probability.

Finally, I should note that these probability distributions can also be trained using [score matching](https://arxiv.org/abs/2011.13456) with varying degrees of success.
Score matching focuses a lot on modelling annealed versions of a PDF which can be way way easier, but may not perform as well at modelling the true (non-annealed) distributions.
