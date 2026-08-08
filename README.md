# Awesome Machine Learning for Graph Drawing

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of papers on **learned graph layout** — neural networks that generate, update, select or optimize the coordinates of a node-link diagram; networks trained to judge the drawings; and the reversal, where a layout is treated as input to a model and optimized for what a machine has to do with it.

Papers are grouped by **what the learned component is trained to reproduce or optimize** — a teacher algorithm's output, an explicit layout objective, a distribution of acceptable layouts, or downstream task utility — rather than by architecture. The same message-passing GNN shows up in three different problem settings below.

## Contents

- [Coverage](#coverage)
- [Layout Generation](#layout-generation)
  - [Algorithm Imitation](#algorithm-imitation)
  - [Explicit Objective Amortization](#explicit-objective-amortization)
  - [Layout Distribution Modeling](#layout-distribution-modeling)
  - [Where No Gradient Reaches the Coordinates](#where-no-gradient-reaches-the-coordinates)
  - [Per-instance Optimization](#per-instance-optimization)
- [Learned Layout Evaluators](#learned-layout-evaluators)
- [Layout for Machine Readers](#layout-for-machine-readers)
- [Benchmarks, Datasets and Evaluation Practice](#benchmarks-datasets-and-evaluation-practice)
- [Foundations and Adjacent Work](#foundations-and-adjacent-work)
- [Scope](#scope)
- [Notes on This List](#notes-on-this-list)
- [Contributing](#contributing)
- [License](#license)

## Coverage

If you came here asking *can I get a neural layout in the style I need?*, this is the answer. The column that matters is not whether a paper exists but whether it leaves you a model you can run on a graph it was not fit to.

| Layout family | Runs on an unseen graph? | What is actually there |
|---|---|---|
| Stress / Kamada-Kawai / MDS | **yes** | DeepGD, GND, GRAPHULY and CoRe-GD each train on a mixed graph corpus and infer on graphs they were not fit to |
| Neighborhood embedding (t-SNE-style) | **yes** | (DNN)² (one graph family) and DL4GD (mixed corpus). (DNN)² is also the only paper here that compares against an optimizer minimizing its own cost, and reports every quality metric except stress significantly worse |
| Prescribed regular patterns (grid, star) | **yes, per style** | DeepDrawing trains one model per style and reproduces a perfect grid and a perfect star |
| Force-directed energy | **weakly evidenced** | Only DeepFD trains across a corpus, and it reports visual quality and runtime rather than a metric-level comparison against the force-directed layouts it takes as its prototype; it also states its training distribution was never analysed. Both et al., Safarli et al. and AutoFDP each optimize one graph at a time and leave no reusable model — and Both et al. replaces the classical long-range repulsion with a short-range Gaussian one, in 3D |
| Layered / Sugiyama | **no** | GL-GRASP trains one model per graph, and its embedding only chooses which vertex the GRASP construction phase places next. Separately, one paper asks ChatGPT to reproduce Sugiyama |
| Circular / radial | **no** | Appears only as a fixed stimulus in a perception benchmark, a library call in an LLM evaluation, a style-transfer target, and a criticised model input |
| Orthogonal | **no** | Excluded by definition in one paper ("we focus on two-dimensional layout methods that draw all edges in straight lines"), listed as future work in another, and the one time it is drawn it is Tamassia's algorithm producing stimuli for machine readers |
| Octilinear / metro-map | **no** | No generator surfaced in a directed search against this absence |
| Clustered / space-filling | **no** | Appears only as input to a metric predictor |

**Where the line falls, and why it is not NP-hardness.** Stress is nonconvex too — the SGD paper below says so about its own guarantee, which "only holds for convergence to a stationary point that may be a saddle." What separates the families above from those below is that stress has a differentiable local search procedure *and* the field accepts its stationary point as the answer, so there is a process to amortize. Crossing-type objectives are discrete: SmartGD, GND and GL-GRASP each state the NP-hardness independently, and AutoFDP names the operative property — these criteria are "challenging due to their discrete and non-convex nature."

**These claims are the fragile part of this list.** They describe this collection plus one directed search aimed at the empty cells. They are claims about what was found, not proofs about the field, and [corrections to them](#contributing) are the most useful contribution here.

## Layout Generation

Entry summaries say what the learned component computes and what it was fit to, rather than what kind of network it is.

### Algorithm Imitation

The learned component reproduces the output of a named layout algorithm. One paper — and note what it takes to make that target well-posed at all: a seeded teacher is nondeterministic, so ForceAtlas2's random initialization is replaced, and the loss compares drawings up to an alignment rather than coordinate by coordinate.

- <details style="display:inline"><summary><b>(<i>TVCG'20</i>) DeepDrawing: A Deep Learning Approach to Graph Drawing</b> [<a href="https://doi.org/10.1109/tvcg.2019.2934798">link</a>] [<a href="https://github.com/jiayouwyhit/deepdrawing">code</a>]</summary> Maps a BFS-ordered adjacency encoding to 2D coordinates with a bidirectional graph-LSTM, trained to reproduce a layout style up to a Procrustes-aligned equivalence class rather than to exact coordinates. One model per style, over ForceAtlas2, PivotMDS, a perfect grid and a perfect star; the ForceAtlas2 teacher is seeded with PivotMDS so that the same graph maps to the same drawing.</details>

### Explicit Objective Amortization

The largest group. The learned component minimizes an objective a classical algorithm also minimizes, but arrives at it by a different computation — so the relationship to graph drawing is a shared loss function, not an imitated output.

- <details style="display:inline"><summary><b>(<i>arXiv'21</i>) Deep Neural Network for DrawiNg Networks, (DNN)²</b> [<a href="https://doi.org/10.48550/arxiv.2108.03632">link</a>] [<a href="https://github.com/LoannGio/DNN2">code</a>]</summary> Amortizes a t-SNE-style layout cost with a ResNet of spectral graph convolutions and no target layout at all. Notably, it runs the comparison almost nobody else runs — against an optimizer minimizing the same cost function — and reports 20.4 ms against 6541 ms, with every quality metric except stress significantly worse.</details>
- <details style="display:inline"><summary><b>(<i>CG&A'21</i>) DeepGD: A Deep Learning Framework for Graph Drawing Using GNN</b> [<a href="https://doi.org/10.1109/mcg.2021.3093908">link</a>] [<a href="https://github.com/yolandalalala/deepgd">code</a>]</summary> A message-passing GNN trained on a weighted combination of stress, minimum angle, edge-length variation, node occlusion and a t-SNE KL term, with two adaptive schemes that rebalance the objectives during training. Infers on unseen graphs of arbitrary topology once trained.</details>
- <details style="display:inline"><summary><b>(<i>TNNLS'22</i>) Graph Neural Networks for Graph Drawing</b> [<a href="https://doi.org/10.1109/tnnls.2022.3184967">link</a>] [<a href="https://github.com/mtiezzi/gnn_graph_draw">code</a>]</summary> Frames drawing as a GNN driven by any loss the user can supply, and tests three kinds: Procrustes distance to ground-truth coordinates from Kamada-Kawai and Spectral, stress, and a learned "Neural Aesthete" approximating edge crossing. Reports that random features were not sufficient for its node-regression task while Laplacian positional encodings were.</details>
- <details style="display:inline"><summary><b>(<i>IEICE'22</i>) GRAPHULY: GRAPH U-Nets-Based Multi-Level Graph LaYout</b> [<a href="https://doi.org/10.1587/transinf.2022edl8050">link</a>]</summary> A Graph U-Net that coarsens and refines, supervised by Kamada-Kawai stress fused across levels and fed Laplacian positional encodings, generating a layout end-to-end without iterative optimization at inference.</details>
- <details style="display:inline"><summary><b>(<i>Nat. Commun.'23</i>) Accelerating network layouts using graph neural networks</b> [<a href="https://doi.org/10.1038/s41467-023-37189-2">link</a>] [<a href="https://github.com/csabath95/NeuLay">code</a>]</summary> Optimizes force-directed energy through the weights of a GCN instead of through coordinates directly, reporting 10–100× speedups against gradient descent on the same energy and tracing the speedup to the spectral structure of the adjacency matrix. States plainly that the model must be retrained for every new graph.</details>
- <details style="display:inline"><summary><b>(<i>ICLR'24</i>) CoRe-GD: A Hierarchical Framework for Scalable Graph Visualization with GNNs</b> [<a href="https://doi.org/10.48550/arxiv.2402.06706">link</a>] [<a href="https://github.com/floriangroetschla/CoRe-GD">code</a>]</summary> Coarsens, lays out and refines with a recurrent GNN, adding positional rewiring so information travels further than the message-passing neighbourhood allows. Names two earlier methods for applying stress directly as a loss and replaces it with a scale-invariant stress with closed-form optimal scaling; proves that minimizing stress demands node distinguishability beyond <i>k</i>-WL for every <i>k</i>, and breaks the symmetry with Laplacian encodings, beacon distances and noise.</details>
- <details style="display:inline"><summary><b>(<i>TVCG'24</i>) Toward Efficient Deep Learning for Graph Drawing (DL4GD)</b> [<a href="https://doi.org/10.1109/tvcg.2022.3222186">link</a>]</summary> Extends (DNN)² and asks what the training data and learning strategy actually buy. Traces the fisheye distortion in its own drawings — centre faithful, periphery contracted, grid corners pulled in — to the cost function itself, noting the effect had been seen before without being connected to that cost.</details>
- <details style="display:inline"><summary><b>(<i>J. Vis.'24</i>) DeepFD: A Deep Learning Approach to Fast Generate Force-Directed Layout for Large Graphs</b> [<a href="https://doi.org/10.1007/s12650-024-00991-1">link</a>]</summary> Proposes rewriting layout methods as loss functions and instantiates one for force-directed energy, training a graph-LSTM encoder and MLP decoder on Louvain subgraphs of very large graphs. Argues explicitly against ground-truth supervision at that scale, on the grounds that generating a target layout per training graph is expensive and its correctness ill-defined. Evaluates on visual quality and runtime, and states that the quality and distribution of its training set were not analysed.</details>

### Layout Distribution Modeling

The target is a set or distribution of acceptable drawings rather than one drawing.

- <details style="display:inline"><summary><b>(<i>TVCG'19</i>) A Deep Generative Model for Graph Layout</b> [<a href="https://doi.org/10.1109/tvcg.2019.2934396">link</a>]</summary> A sliced-Wasserstein autoencoder trained on 20,000 layouts of a single graph, producing a 2D latent space a user can traverse to generate different drawings. The only mechanism in this list for asking an already-trained model for another layout — and it is one model per graph, with diversity bounded by the four algorithms and parameter sweep that generated its training set.</details>
- <details style="display:inline"><summary><b>(<i>TVCG'24</i>) SmartGD: A GAN-Based Graph Drawing Framework for Diverse Aesthetic Goals</b> [<a href="https://doi.org/10.1109/tvcg.2023.3306356">link</a>] [<a href="https://github.com/yolandalalala/SmartGD">code</a>]</summary> A conditional relativistic GAN that learns from a pool of good layout examples and, through a self-challenging mechanism, replaces those examples with better ones it generated itself. Optimizes non-differentiable criteria — crossings, crossing angle — without hand-building a differentiable surrogate; an ablation runs with no example layouts at all.</details>

### Where No Gradient Reaches the Coordinates

Reinforcement learning over node moves, a force model synthesized from stated criteria, and a learned embedding supplying one decision inside a metaheuristic. None has an end-to-end gradient from a layout objective to the coordinates, and none leaves a model that runs on another graph — all three optimize one instance at a time.

- <details style="display:inline"><summary><b>(<i>arXiv'20</i>) Interpreting Graph Drawing with Multi-Agent Reinforcement Learning</b> [<a href="https://doi.org/10.48550/arxiv.2011.00748">link</a>]</summary> One agent per node, each observing a 3×3 local grid and choosing among nine moves. Recovers force-directed layout and stress majorization as reward maximization, and proposes designing new drawing algorithms by designing new rewards. Optimized per graph instance; the claim is that classical algorithms can be *interpreted* this way, not that a model learned them.</details>
- <details style="display:inline"><summary><b>(<i>TVCG'25</i>) AutoFDP: Automatic Force-Based Model Selection for Multicriteria Graph Drawing</b> [<a href="https://doi.org/10.1109/TVCG.2025.3631659">link</a>] [<a href="https://github.com/IDEASLab-AutoFDP/AutoFDP">code</a>]</summary> Searches the six parameters of a quotient-based force model against user-chosen readability criteria, so the output is a force model rather than coordinates, and supports different criteria on different subgraphs. Non-neural bi-level optimization. Reports that a fitted model transfers to larger unseen graphs when topological kernel similarity is at least 0.5.</details>
- <details style="display:inline"><summary><b>(<i>EJOR'25</i>) An efficient hybridization of Graph Representation Learning and metaheuristics for the Constrained Incremental Graph Drawing Problem</b> [<a href="https://doi.org/10.1016/j.ejor.2025.08.034">link</a>] [<a href="https://github.com/bcbraga/CIGDP-DL">code</a>]</summary> Node embeddings decide which vertex the GRASP construction phase places next; the coordinates come from the metaheuristic. Improves on hand-built GRASP in a combined time-and-quality measure and holds up on denser instances. This is the worked boundary case that defines the [scope test](#scope).</details>

### Per-instance Optimization

Not a method family — a reference point, and the procedure the amortized methods amortize.

- <details style="display:inline"><summary><b>(<i>TVCG'18</i>) Graph Drawing by Stochastic Gradient Descent</b> † [<a href="https://doi.org/10.1109/TVCG.2018.2859997">link</a>] [<a href="https://github.com/jxz12/s_gd2">code</a>]</summary> Minimizes stress by moving one pair of vertices at a time, reaching lower stress faster and more consistently than majorization and without needing a good initialization; extends naturally to constrained layout and to a sparse approximation for large graphs. States that for a nonconvex function like stress, a convergence proof reaches only a stationary point that may be a saddle.</details>

† Not a neural method, kept here because it is the procedure the amortized methods amortize.

## Learned Layout Evaluators

Networks trained to score, rank or predict the quality of a drawing. Ordered by where the labels came from, which moves in one direction over time: computed metrics, then a layout algorithm used as a proxy for quality, then metric pretraining fine-tuned on human labels, then human labels directly, and most recently models standing in for the human annotators themselves. The input moves with it, from graph structure with nothing rendered, to rendered images, to both. These evaluators make a generator–critic loop structurally available, and no generator in this list is trained by one; only SmartGD is trained adversarially.

- <details style="display:inline"><summary><b>(<i>TVCG'18</i>) What Would a Graph Look Like in this Layout? A Machine Learning Approach to Large Graph Visualization</b> [<a href="https://doi.org/10.1109/tvcg.2017.2743858">link</a>] [<a href="https://github.com/kwonoh/glam">code</a>]</summary> Learns topological similarity between graphs so a layout's appearance and its aesthetic metrics can be previewed without running each layout algorithm. The input is graph structure alone — nothing is rendered — and the target is layout selection for large graphs rather than coordinates. Restricts itself by definition to methods that draw all edges as straight lines.</details>
- <details style="display:inline"><summary><b>(<i>CG&A'19</i>) Evaluating the Readability of Force Directed Graph Layouts: A Deep Learning Approach</b> [<a href="https://doi.org/10.1109/mcg.2018.2881501">link</a>]</summary> A CNN predicting readability metrics — edge crossings, node spread, group overlap — directly from a rendered layout image, so quality can be quantified when node and edge coordinates are unavailable. Reports high percentage error precisely where the 2D image is ambiguous: node occlusion and group overlap.</details>
- <details style="display:inline"><summary><b>(<i>JGAA'19</i>) Aesthetic Discrimination of Graph Layouts</b> [<a href="https://doi.org/10.7155/jgaa.00501">link</a>] [<a href="https://github.com/5gon12eder/msc-graphstudy">code</a>]</summary> Trains a discriminator to pick the more aesthetic of two drawings of the same graph from a feature vector fusing classical metrics, statistics, information-theoretic quantities and two-point correlations. Labels come from which layout algorithm produced the drawing, not from people; the journal version adds a larger paired dataset and augmentation.</details>
- <details style="display:inline"><summary><b>(<i>JGAA'22</i>) A Machine Learning Approach for Predicting Human Preference for Graph Drawings</b> [<a href="https://doi.org/10.7155/jgaa.00603">link</a>]</summary> A Siamese CNN predicting which of two drawings a person prefers, pretrained on quality metrics correlated with human preference and fine-tuned on experimental labels to work around the scarcity of human data. Also records symmetry-faithful metrics — whether a graph's automorphisms actually appear as symmetries in the drawing.</details>
- <details style="display:inline"><summary><b>(<i>J. Vis.'25</i>) A deep learning approach to evaluate the quality of graph layouts using GNN</b> [<a href="https://doi.org/10.1007/s12650-025-01044-x">link</a>]</summary> Predicts layout quality from node coordinates and edge attributes through graph convolutions and pooling rather than from a rendered image, keeping the topology and geometry that rasterization discards. Reports substantial training-time overhead above roughly 10,000 nodes.</details>
- <details style="display:inline"><summary><b>(<i>CGF'26</i>) Beauty in the Eye of AI: Aligning LLMs and Vision Models with Human Aesthetics in Network Visualization</b> [<a href="https://doi.org/10.1111/cgf.70456">link</a>]</summary> Asks whether LLMs and vision models can serve as scalable stand-ins for human annotators of layout preference, with a 27-participant study as the reference. Few-shot examples, input format and confidence filtering move LLM–human agreement substantially, and a purpose-trained vision model approaches the agreement level between human annotators.</details>

## Layout for Machine Readers

The newest branch, and the one that inverts the premise. A drawing stops being the human-facing artifact and becomes model input — then becomes something optimized backward from what the model has to do with it. The consequence for everything above is direct: the aesthetic criteria those loss functions encode were established in human studies, and they do not transfer unchanged.

### Do Models Read Drawings the Way People Do

- <details style="display:inline"><summary><b>(<i>CGF'26</i>) Do Graph Drawing Aesthetics Matter for AI? A Replication of Foundational Studies in Graph Readability</b> [<a href="https://doi.org/10.1111/cgf.70436">link</a>]</summary> Repeats the foundational readability experiments — edge crossings, symmetry, bends — with AI models in place of human participants, and finds the preferences established for human readers do not carry over unchanged; minimizing crossings does not reliably help, while symmetry matters more. Also the only paper in this list that actually draws orthogonal layouts, using Tamassia's planar orthogonal grid algorithm, and it draws them as stimuli.</details>
- <details style="display:inline"><summary><b>(<i>arXiv'26</i>) Seeing Graphs Like Humans: Benchmarking Computational Measures and MLLMs for Similarity Assessment</b> [<a href="https://doi.org/10.48550/arXiv.2602.22416">link</a>] [<a href="https://github.com/skwn-j/gsa-dataset">code</a>]</summary> Builds a perceptual similarity benchmark from 1,881 node-link drawings and 32 participants, then compares 16 computational measures and three MLLMs against it. The MLLMs track human judgement more closely than the measures do, and can state a reason for the judgement.</details>

### Benchmarks for Visual Graph Reasoning

- <details style="display:inline"><summary><b>(<i>NeurIPS'24</i>) GITA: Graph to Visual and Textual Integration for Vision-Language Graph Reasoning</b> [<a href="https://doi.org/10.52202/079017-0002">link</a>] [<a href="https://github.com/WEIYanbin1999/GITA">code</a>]</summary> Pairs a rendered image of a graph with its textual description for general graph reasoning, and releases GVLQA, a vision-language QA dataset for it. Reports that visual input, layout augmentation and dataset pretraining all improve reasoning on real graph tasks.</details>
- <details style="display:inline"><summary><b>(<i>ICML'25</i>) Visual Graph Arena: Evaluating Visual Conceptualization of Vision and Multimodal Large Language Models</b> [<a href="https://doi.org/10.48550/arxiv.2506.06242">link</a>]</summary> Six task families rendered under different layouts — Kamada-Kawai, planar — to test whether a model reasons about the graph or about the picture. Humans are near-perfect; models are not, and the failures on isomorphism, paths and cycles point at surface pattern matching rather than layout-invariant understanding.</details>

### Layouts Optimized for a Downstream Machine Task

- <details style="display:inline"><summary><b>(<i>WWW'26</i>) VSAL: A Vision Solver with Adaptive Layouts for Graph Property Detection</b> [<a href="https://doi.org/10.1145/3774904.3792224">link</a>] [<a href="https://github.com/Jiahao-Xie-86/VSAL">code</a>]</summary> Jointly learns the layout and the image classifier for visual graph property detection — Hamiltonian cycle, planarity, claw-freeness, tree recognition — so the drawing emphasizes what the model must discriminate rather than what a person would find readable. The only method in this list with a gradient path from a downstream task back to the coordinates, through a differentiable renderer.</details>

### Language Models Asked to Produce or Judge Layouts

- <details style="display:inline"><summary><b>(<i>EuroVis'23</i>) Ask and You Shall Receive (a Graph Drawing): Testing ChatGPT's Potential to Apply Graph Layout Algorithms</b> [<a href="https://doi.org/10.2312/evs.20231047">link</a>]</summary> An early probe of whether a language model can apply layout algorithms when asked, including Sugiyama-style layered drawing, and of what probabilistic generation costs in algorithmic correctness. The natural-language interface to graph drawing, stated as an opportunity with its risk attached.</details>
- <details style="display:inline"><summary><b>(<i>Vis. Inform.'26</i>) How well will LLMs perform for graph layout tasks?</b> [<a href="https://doi.org/10.1016/j.visinf.2025.100285">link</a>]</summary> Evaluates GPT-4o, Gemini 2.0 and DeepSeek-V3 across three separable abilities — understanding a graph, generating a layout, and evaluating one — over varying graph types, sizes, serialization formats and prompting modes, mapping where the capability boundary currently falls.</details>

## Benchmarks, Datasets and Evaluation Practice

- <details style="display:inline"><summary><b>(<i>CGF'24</i>) Evaluating Graph Layout Algorithms: A Systematic Review of Methods and Best Practices</b> [<a href="https://doi.org/10.1111/cgf.15073">link</a>]</summary> Reviews how layout algorithms are actually evaluated — computational metrics, benchmark selection, user studies — and finds no common standard. Summarizes the trade-offs of each evaluation strategy and collects the available resources.</details>
- <details style="display:inline"><summary><b>(<i>GD'23</i>) A collection of benchmark datasets for evaluating graph layout algorithms</b> [<a href="https://doi.org/10.31219/osf.io/yftju">link</a>]</summary> A collected set of benchmark graphs for layout evaluation, with a website for retrieval and supplementary material. Its value is comparability across studies that would otherwise each pick their own graphs.</details>

## Foundations and Adjacent Work

### Expressivity Results This List Reasons With

Not objects of this list. They are here because they are theorems about the architecture graph drawing runs on — equivariant models collapsing automorphism orbits, message passing being local while stress is global — and GND and CoRe-GD both reach for them, citing random features and random node initialization as a *device* for making nodes distinguishable. **No paper above is described as knowing or citing this literature unless its own text does so.** The neural algorithmic reasoning / ML4CO side cites the same upstream for the opposite purpose, raising a proven expressivity ceiling; for that, see [awesome-neural-algorithmic-reasoning-and-combinatorial-optimization](https://github.com/zhangleiniu/awesome-neural-algorithmic-reasoning-and-combinatorial-optimization).

- <details style="display:inline"><summary><b>(<i>NeurIPS'19</i>) Approximation Ratios of Graph Neural Networks for Combinatorial Problems</b> [<a href="https://doi.org/10.48550/arxiv.1905.10261">link</a>]</summary> Shows the graph problems a GNN can solve coincide with those solvable by distributed local algorithms, so approximation-ratio upper bounds are inherited from distributed lower bounds — and that under a fixed architecture, changing only the node features changes the best achievable ratio.</details>
- <details style="display:inline"><summary><b>(<i>SDM'21</i>) Random Features Strengthen Graph Neural Networks</b> [<a href="https://doi.org/10.1137/1.9781611976700.38">link</a>] [<a href="https://github.com/joisino/random-features">code</a>]</summary> Adding random features to nodes provably increases discriminative power. Cited on the layout side as a way to break symmetry between nodes, and on the combinatorial side as a way to repair an impossibility result.</details>
- <details style="display:inline"><summary><b>(<i>IJCAI'21</i>) The Surprising Power of Graph Neural Networks with Random Node Initialization</b> [<a href="https://doi.org/10.24963/ijcai.2021/291">link</a>]</summary> Random node initialization takes GNNs past the 1-WL discrimination bound. The other upstream paper both literatures cite.</details>
- <details style="display:inline"><summary><b>(<i>ICLR'23</i>) On Representing Mixed-Integer Linear Programs by Graph Neural Networks</b> [<a href="https://doi.org/10.48550/arxiv.2210.10759">link</a>] [<a href="https://github.com/liujl11git/GNN-MILP">code</a>]</summary> Proves the variable–constraint bipartite encoding cannot separate some feasible from infeasible MILPs, and repairs it with random features. Its hypothesis is about a problem class, so the conclusion does not travel to layout.</details>
- <details style="display:inline"><summary><b>(<i>JMLR'23</i>) Combinatorial Optimization and Reasoning with Graph Neural Networks</b> [<a href="https://www.jmlr.org/papers/v24/21-0449.html">link</a>]</summary> The survey of GNNs for combinatorial optimization and reasoning. It is also the single substantive citation crossing from the layout literature into that one, in VSAL's reference list. There is an earlier, shorter <a href="https://doi.org/10.24963/ijcai.2021/595">IJCAI 2021 survey-track version</a>; the JMLR article is the one read here and the one labelled.</details>

### Adjacent

Neighbouring work, listed for orientation rather than covered. In each case either no learned component touches the drawing procedure, or no drawing is produced at all.

- <details style="display:inline"><summary><b>(<i>TVCG'23</i>) Graph Exploration With Embedding-Guided Layouts</b> [<a href="https://doi.org/10.1109/TVCG.2023.3238909">link</a>] [<a href="https://github.com/tzw28/EmbeddingGuidedLayout">code</a>]</summary> Encodes topology and node attributes into an embedding to produce layouts that preserve communities, then builds aggregation, focus+context and related-node search around them. <b>The worked "out" case for the scope test:</b> the learned component rewrites the input graph handed to an unmodified Fruchterman-Reingold, leaving the drawing procedure itself untouched. Its contribution is the exploration workflow.</details>
- <details style="display:inline"><summary><b>(<i>TVCG'24</i>) GVVST: Image-Driven Style Extraction From Graph Visualizations for Visual Style Transfer</b> [<a href="https://doi.org/10.1109/TVCG.2024.3485701">link</a>]</summary> Extracts global and local visual style — colour, layout, node and edge rendering — from a bitmap of a graph visualization and transfers it to a new one. Inverse visualization and design reuse rather than layout generation.</details>
- <details style="display:inline"><summary><b>(<i>TVCG'22</i>) A Survey on ML4VIS: Applying Machine Learning Advances to Data Visualization</b> [<a href="https://doi.org/10.1109/tvcg.2021.3106142">link</a>]</summary> Organizes 88 ML4VIS works along two axes — the visualization stages machine learning can assist, and the learning tasks involved. Useful for locating learned graph layout within the wider ML4VIS map.</details>
- <details style="display:inline"><summary><b>(<i>TVCG'24</i>) Visual Analytics of Multivariate Networks With Representation Learning and Composite Variable Construction</b> [<a href="https://doi.org/10.1109/tvcg.2024.3423728">link</a>] [<a href="https://github.com/hylu1994/Network-CV">code</a>]</summary> Uses representation learning and composite variable construction to help interpret structure–attribute relationships in multivariate networks, turning nonlinear neural features back into interpretable variables. Network analysis, not layout generation.</details>
- <details style="display:inline"><summary><b>(<i>CVPR'18</i>) Image Generation From Scene Graphs</b> [<a href="https://doi.org/10.1109/cvpr.2018.00133">link</a>] [<a href="https://github.com/google/sg2im">code</a>]</summary> Generates images from scene graphs by encoding object relations with graph convolutions, predicting an object layout, and refining it into an image. Not a graph drawing paper; included as the neighbouring paradigm for "graph structure to visual layout".</details>

## Scope

A paper is in when **a learned component generates, updates, selects or optimizes the drawing itself, or judges one**. Producing an embedding that some drawing procedure later consumes is not sufficient — the question is whether the learned component participates in producing the drawing. Two worked decisions say more than the rule does:

- **In —** [GL-GRASP](#where-no-gradient-reaches-the-coordinates): the embedding determines which vertex the GRASP construction phase places next. That decision is part of producing the drawing, even though the coordinates come out of the metaheuristic.
- **Out —** [GEGraph](#adjacent): the learned component rewrites the input graph that is then handed to an unmodified Fruchterman-Reingold. The drawing procedure is untouched, so what is learned is an input, not a drawing.

**Layout family is held distinct from graph family throughout.** These papers change their input graph distributions constantly — trees, grids, planar graphs, community graphs — and almost never change their drawing convention. Reading "evaluated on tree datasets" as "covers tree layouts" is the most likely way to misuse the [coverage table](#coverage).

## Notes on This List

- **Every entry has been read**, and its summary written from the paper's own text. Reading depth varies: the layout-generation papers were read at passage depth or deeper; several evaluation and machine-reading entries at abstract-plus-results depth.
- **`[code]` links were verified, and their absence is informative.** One was added only where the paper's own text names the repository, or where the repository identifies itself as that paper's implementation and is maintained by an author. Third-party reimplementations, repositories a paper cites as dependencies, project websites and anonymized submission links were all rejected, and one paper's repository was not carried to its own successor: `LoannGio/DNN2` names the (DNN)² paper and not DL4GD. **No `[code]` means none was found under that rule, not that none exists.**
- **Venue labels name the venue of record**, taken from a bibliography checked against the PDFs. Link text is `link` because the URL points at the paper's page, which is not always a PDF.
- **No PDFs are hosted here.** Open access is not a licence to redistribute.

## Contributing

Issues and pull requests welcome, for missing papers, miscategorised ones, and summaries that misstate what a paper claims.

**Corrections to [Coverage](#coverage) are the most valuable kind.** That table asserts that certain layout families have no generator that runs on an unseen graph. A single counterexample changes it, and a counterexample is more useful here than a new entry anywhere else on the page.

Entries use a collapsible summary, so the one-liner stays hidden until the title is clicked:

```html
- <details style="display:inline"><summary><b>(<i>Venue'YY</i>) Paper title</b> [<a href="paper-url">link</a>] [<a href="code-url">code</a>]</summary> One-line summary.</details>
```

The summary line is raw HTML, so use `<b>` / `<i>` / `<a>` rather than Markdown syntax inside it.

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, this list is released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/) — public domain. Cited papers remain under their own copyright.
