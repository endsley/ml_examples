<TeXmacs|1.0.7.18>

<style|generic>

<\body>
  <doc-data|<doc-title|Deep Spectral Clustering>|<doc-author|<\author-data|<author-name|Chieh,
  Khan>>
    \;
  </author-data>>>

  Deep learning has achieved some of the most amazing results in supervised
  machine learning. It does suffer from some minor issues, e.g.
  interpretability, require a lot of data, overfitting, etc. In spite of
  these problems, it regularly achieve superior results in the supervised
  setting. \ 

  Changing the domain into unsupervised settings, namely in clustering
  problems the benefits of utilizing neural networks is less obvious.
  Historically, clustering in neural networks has been performed through the
  strategy of combining dimensionality reduction and clustering. This is a
  well established approach in classical clustering in that we often perform
  PCA/LLE to reduce the dimensionality of the data prior to some clustering
  algorithm. In neural networks, this dimensionality reduction is done
  through the usage of autoencoders.\ 

  \;

  In 2012, in the paper by Baldi :

  \ \ \ \ \ Autoencoders, unsupervised \ Learning and Deep Architectures\ 

  The paper discussed the connection between autoencoders and clustering.\ 

  \;

  In 2013, in the paper by Song et al :

  \ \ \ \ \ Autoencoder based Data Clustering

  The paper proposed to 1st perform autoencoder and then some sort of
  clustering algorithm.

  \;

  In 2014, in the paper by Tian et al :

  \ \ \ \ \ Learning Deep Representation for Graph Clustering

  The paper provided theretical reasoning why autoencoders + Kmeans is better
  than spectral clustering.

  \;

  In 2015, in the paper by Chen :\ 

  \ \ \ \ \ Deep Learning with Non-parametric clustering

  The paper used DBN instead of autoencoders to perform dimensionality
  reduction. Once in the feature speace, they designed a graphical model, and
  used Gibb's sampling to get the posterior.\ 

  \;

  The idea of performing clustering after dimesionality reduction is such a
  dominant strategy, that even at 2017, in the paper by Dilokthanakul.

  \ \ \ \ \ Deep unsupervised clustering with Gaussian mixture auto encoders

  This paper is still using the idea of autoencoders.\ 

  \;

  There was 1 paper in 2016 ICML that deviated from autoencoders by Xie et al
  :

  \ \ \ \ \ Unsupervised Deep embedding for clustering analysis

  Instead of using autoencoders, they used a self-training approach. Here are
  the steps of their algorithm.

  1. Perform regular clustering

  2. Use the most confident data sets to model a stribution <math|p>

  3. Use DNN to discover a mapping <math|f<rsub|\<theta\>>> from data to
  feature space with a distribution <math|q> such that the
  <math|KL<around*|(|p<around*|\|||\|>q|)>> is minimized.\ 

  4. Repeat this process with more data.\ 

  \;

  The idea of this paper is to use the most confident points as an
  approximation to the truth data distribution. \ Given an approximation, DNN
  is used to bias the data towards the distribution of the most confident
  data sents. This type of self training strategy allows for the neural
  network to optimize an objective function of
  KL<math|<around*|(|p<around*|\|||\|>q|)>>.\ 

  \;

  As observed, the idea of performing some sort of dimensionality reduction
  prior to clustering has been the predominant strategy. Since autoencoders
  perform dimensionality reduction for neural networks, its usage has been
  obvious. However, as we know from performing PCA prior to clustering. The
  compression of data doesn't always yield a good clustering. Instead, it is
  better to discover a lower dimensional representation of the data that are
  contrained simultaneously to yield a good clustering result.\ 

  To achieve this objective, we introduce the following formulation.\ 

  <\equation*>
    <tabular|<tformat|<cwith|2|2|1|1|cell-halign|c>|<cwith|1|1|1|1|cell-halign|c>|<table|<row|<cell|max>>|<row|<cell|U\<nocomma\>,\<varphi\><around*|(|.|)>>>>>>\<cal-R\><around*|(|\<varphi\><around*|(|X;W|)>,U|)>
    <tabular|<tformat|<table|<row|<cell|>|<cell|>|<cell|s.t>|<cell|>|<cell|>>>>>U<rsup|T>U=I
  </equation*>

  Let <math|\<cal-R\><around*|(|A,B|)>> be a function that measures the
  relationship between two datasets A and B. This function could be replaced
  by any similarity measure such as mutual information, correlation, or HSIC.
  Let the input data be <math|X\<in\>\<bbb-R\><rsup|n\<times\>d>>, where
  <math|n> is the number of samples and <math|d> the number of features. Let
  <math|U> be the corresponding clustering of <math|X>. The formulation above
  wishes to discover a mapping of <math|X> and a clustering solution <math|U>
  such that the relationship between <math|\<varphi\><around*|(|X|)>> and
  <math|U> is maximized. We let <math|\<varphi\><around*|(|X;W|)>> be a
  neural network with <math|W> as all of its weights. The optimization
  process therefore iterates between optimizing the weights of the neural
  networks and the clustering <math|U>.

  For the purpose of this paper, HSIC was used as a relationship measure. (
  List the reasons why HSIC is an ideal choice ). Therefore the formulation
  could be rewritten as\ 

  <\equation*>
    <tabular|<tformat|<cwith|2|2|1|1|cell-halign|c>|<cwith|1|1|1|1|cell-halign|c>|<table|<row|<cell|max>>|<row|<cell|U\<nocomma\>,\<varphi\><around*|(|.|)>>>>>>HSIC<around*|(|\<varphi\><around*|(|X;W|)>,U|)>=<tabular|<tformat|<cwith|2|2|1|1|cell-halign|c>|<cwith|1|1|1|1|cell-halign|c>|<table|<row|<cell|max>>|<row|<cell|U\<nocomma\>,\<varphi\><around*|(|.|)>>>>>>Tr<around*|(|K<rsub|\<varphi\><around*|(|X;W|)>>H
    K<rsub|U>H|)>.
  </equation*>

  Where <math|H> is the centering matrix, <math|K<rsub|U>> is a linear Kernel
  of <math|U> where <math|K<rsub|U>=U U<rsup|T>> and
  <math|K<rsub|\<varphi\><around*|(|X;W|)>>> is another kernel perform on the
  data after <math|\<varphi\><around*|(|.|)>>. Since <math|U> represents
  labels, it is reasonable to use the linear kernel to represent this data
  type. However, the kernel suitable for <math|K<rsub|\<varphi\><around*|(|X;W|)>>>
  depends on the data itself. To prevent the explosion of the objective
  function, a Gaussian kernel was used to normalize the data between 0 to 1.\ 

  Assuming that the kernels are centered, the objective function can written
  in the formulation of spectral clustering.\ 

  <\equation*>
    Tr<around*|(|K<rsub|\<varphi\><around*|(|X;W|)>>U
    U<rsup|T>|)>=Tr<around*|(|U<rsup|T> K<rsub|\<varphi\><around*|(|X;W|)>>U
    |)>
  </equation*>

  Therefore, there is a strong relationship between finding a high HSIC
  relationship and a good clustering quality in the spectral clustering
  sense. In spectral clustering, the concept of normalized Kernel allows for
  a better clustering results. The kernel is normalized by dividing by the
  Degree Matrix <math|D>. Adding the Degree matrix into the formulation, the
  new objective function becomes.\ 

  <\equation*>
    <tabular|<tformat|<table|<row|<cell|<tabular|<tformat|<cwith|1|1|1|1|cell-halign|c>|<cwith|2|2|1|1|cell-halign|c>|<table|<row|<cell|max>>|<row|<cell|\<varphi\><around*|(|X;W|)>,U>>>>>>|<cell|Tr<around*|(|
    D<rsup|-1/2>K<rsub|\<varphi\><around*|(|X;W|)>>D<rsup|-1/2> H K<rsub|U>
    H|)>>>|<row|<cell|s.t>|<cell|U<rsup|T>U=I>>>>>
  </equation*>

  Because this is a highly non-convex problem, the proper initalization point
  heavily determines the both the quality of the clustering and the speed of
  convergence.\ 
</body>

<\initial>
  <\collection>
    <associate|language|american>
    <associate|page-type|letter>
  </collection>
</initial>